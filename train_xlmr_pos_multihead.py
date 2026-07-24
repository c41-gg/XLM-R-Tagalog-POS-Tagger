"""
Multi-head XLM-RoBERTa fine-tuning for Tagalog-English POS tagging on the
MGNN tagset, factored into 5 orthogonal heads (see modules/mgnn_schema.py),
ALL single-label classification:

    category  -- coarse POS
    subtype   -- fine-grained type within category (null class when N/A)
    focus     -- verb focus marker                 (null class when N/A)
    degree    -- adjective degree marker            (null class when N/A)
    extra     -- attached CCP/CCA/LM/PRSP/CCB tag   (null class when none)

"extra" is single-label, not multi-label: across every token checked so far,
none carry more than one attached tag at once. If your data ever does
produce >1, this script takes the first (alphabetically) and logs a count --
see the "[extra] token had N>1 extras" warning at data-load time. If you see
that warning fire non-trivially, multi-label is the more correct model and
this script would need to go back to a sigmoid head.

Every gold label is produced by calling YOUR modules/mgnn_schema.decompose()
directly.

--------------------------------------------------------------------------
Label maps (REQUIRED -- one JSON file per head)
--------------------------------------------------------------------------
--category_map / --subtype_map / --focus_map / --degree_map / --extra_map
each point to a flat {tag: id} JSON file, e.g.:

    {"__NULL__": 0, "NNC": 1, "NNP": 2, ...}

The script auto-detects whichever "no value" sentinel key you used per file
(recognizes NONE / __NONE__ / NULL / __NULL__ / UNK / __UNK__) rather than
assuming a fixed name -- but every axis MUST have exactly one such sentinel
already present, except category, where one is added automatically as
"__UNK__" with a warning if none exists (yours currently has none).

--------------------------------------------------------------------------
Mandatory pre-flight audit
--------------------------------------------------------------------------
Before any tokenization happens, every file you pass via --train_files/
--val_files/--test_files is fully decomposed and every value actually
produced by decompose() on every axis is diffed against your provided maps.
If anything appears in the data that has no id in your map (e.g. a stray
PRSP extra with no matching entry), the script HALTS with the exact list of
missing values and their counts -- training does not start silently on
partially-unmapped labels. Pass --allow_unmapped_labels to proceed anyway;
missing values then fall back to that axis's null/unk sentinel, and the
count of how many tokens that affected is printed so you know the damage.

Usage:
    python train_xlmr_pos_multihead.py `
        --train_files data/processed/train.jsonl `
        --val_files   data/processed/validation.jsonl `
        --test_files  data/processed/test.jsonl `
        --category_map inferrence/category_map.json `
        --subtype_map  inferrence/subtype_map.json `
        --focus_map    inferrence/focus_map.json `
        --degree_map   inferrence/degree_map.json `
        --extra_map    inferrence/extra_map.json `
        --schema_dir . `
        --output_dir runs/xlmr-tl-en-pos-multihead `
        --epochs 5 `
        --batch_size 4 --gradient_accumulation_steps 4 `
        --eval_batch_size 4 `
        --gradient_checkpointing --fp16


Requires: transformers, torch, scikit-learn
    pip install transformers torch scikit-learn --break-system-packages
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    set_seed,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
 
 
NULL_ALIASES = ["NONE", "__NONE__", "NULL", "__NULL__"]
UNK_ALIASES = ["UNK", "__UNK__", "UNKNOWN"]
AXES = ["category", "subtype", "focus", "degree", "extra"]
 
 
# ---------------------------------------------------------------------------
# Robust JSONL loading (handles lines with multiple JSON objects glued
# together with no separating newline -- seen in earlier pipeline output)
# ---------------------------------------------------------------------------
 
_decoder = json.JSONDecoder()
 
 
def _iter_json_objects(line):
    idx, n = 0, len(line)
    while idx < n:
        while idx < n and line[idx] in " \t":
            idx += 1
        if idx >= n:
            break
        obj, end = _decoder.raw_decode(line, idx)
        yield obj
        idx = end
 
 
def load_jsonl_examples(paths):
    examples = []
    n_lines = n_skipped_parse = n_skipped_shape = n_skipped_none = 0
 
    for path in paths:
        with open(Path(path), encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                n_lines += 1
                try:
                    objs = list(_iter_json_objects(raw_line))
                except json.JSONDecodeError:
                    n_skipped_parse += 1
                    continue
                for obj in objs:
                    tokens = obj.get("tokens")
                    labels = obj.get("labels")
                    if not tokens or not labels or len(tokens) != len(labels):
                        n_skipped_shape += 1
                        continue
                    if any(l is None for l in labels):
                        n_skipped_none += 1
                        continue
                    examples.append({"tokens": tokens, "labels": labels})
 
    print(f"[data] read {n_lines} lines from {len(paths)} file(s) -> {len(examples)} usable sentences")
    if n_skipped_parse:
        print(f"[data] skipped {n_skipped_parse} unparseable lines")
    if n_skipped_shape:
        print(f"[data] skipped {n_skipped_shape} records with missing/mismatched tokens+labels")
    if n_skipped_none:
        print(f"[data] skipped {n_skipped_none} records containing a null label")
    return examples
 
 
# ---------------------------------------------------------------------------
# Decompose every gold label once, up front, via the real schema module.
# ---------------------------------------------------------------------------
 
def decompose_dataset(examples, decompose_fn):
    decomposed_sentences = []
    n_extra_subtype_hits = 0
    n_unknown_part_hits = 0
    n_multi_extra_hits = 0
    offending = Counter()
 
    for ex in examples:
        sent_decomp = []
        for label in ex["labels"]:
            d = decompose_fn(label)
            if d.extra_subtypes:
                n_extra_subtype_hits += 1
                offending[label] += 1
            if d.unknown_parts:
                n_unknown_part_hits += 1
                offending[label] += 1
            if len(d.extras) > 1:
                n_multi_extra_hits += 1
            sent_decomp.append(d)
        decomposed_sentences.append(sent_decomp)
 
    if n_extra_subtype_hits or n_unknown_part_hits:
        print(f"[schema] {n_extra_subtype_hits} token(s) had genuine same-axis "
              f"stacking (extra_subtypes) -- subtype head keeps only the PRIMARY subtype")
        print(f"[schema] {n_unknown_part_hits} token(s) had a part not in ATOMIC_TAGS")
        print(f"[schema] top offending raw labels: {offending.most_common(10)}")
    if n_multi_extra_hits:
        print(f"[extra] {n_multi_extra_hits} token(s) had >1 extra tag simultaneously -- "
              f"the single-label 'extra' head keeps only the first (alphabetically). "
              f"If this number is large relative to your corpus, switch to a multi-label "
              f"sigmoid head instead -- this script assumes it's negligible.")
 
    return decomposed_sentences
 
 
# ---------------------------------------------------------------------------
# Label maps: load user-provided maps, auto-detect their null/unk sentinel
# ---------------------------------------------------------------------------
 
def find_sentinel(label_map, aliases, axis_name, add_if_missing=False, add_name="__UNK__"):
    for alias in aliases:
        if alias in label_map:
            return alias
    if add_if_missing:
        new_id = max(label_map.values(), default=-1) + 1
        label_map[add_name] = new_id
        print(f"[labels] '{axis_name}' map has no null/unk sentinel -- added '{add_name}' at id {new_id}")
        return add_name
    raise ValueError(
        f"'{axis_name}' map has none of the recognized null sentinels {aliases}. "
        f"Add one (e.g. \"__NULL__\": <id>) before running."
    )
 
 
def load_all_label_maps(args):
    maps = {}
    sentinels = {}
    for axis, path in [("category", args.category_map), ("subtype", args.subtype_map),
                        ("focus", args.focus_map), ("degree", args.degree_map),
                        ("extra", args.extra_map)]:
        with open(path, encoding="utf-8") as f:
            maps[axis] = json.load(f)
 
    sentinels["category"] = find_sentinel(maps["category"], NULL_ALIASES + UNK_ALIASES,
                                           "category", add_if_missing=True)
    for axis in ["subtype", "focus", "degree", "extra"]:
        sentinels[axis] = find_sentinel(maps[axis], NULL_ALIASES, axis, add_if_missing=False)
 
    for axis in AXES:
        print(f"[labels] {axis}: {len(maps[axis])} classes (null/unk sentinel = '{sentinels[axis]}')")
 
    return maps, sentinels
 
 
# ---------------------------------------------------------------------------
# Mandatory audit: every value decompose() produces vs. what the maps cover
# ---------------------------------------------------------------------------
 
def audit_label_coverage(all_decomposed, maps, sentinels, allow_unmapped=False):
    seen = {axis: Counter() for axis in AXES}
 
    for sent in all_decomposed:
        for d in sent:
            seen["category"][d.category] += 1
            seen["subtype"][d.subtype] += 1
            seen["focus"][d.focus] += 1
            seen["degree"][d.degree] += 1
            if d.extras:
                for e in sorted(d.extras)[:1]:  # single-label: keep first only
                    seen["extra"][e] += 1
            else:
                seen["extra"][None] += 1
 
    any_missing = False
    for axis in AXES:
        provided_keys = set(maps[axis].keys()) - {sentinels[axis]}
        seen_values = {k: v for k, v in seen[axis].items() if k is not None}
        missing = {k: v for k, v in seen_values.items() if k not in provided_keys}
        unused = provided_keys - set(seen_values.keys())
        if missing:
            any_missing = True
            print(f"[audit] {axis}: MISSING FROM MAP -> {missing}  "
                  f"(these tokens have no id -- {'will fall back to sentinel' if allow_unmapped else 'BLOCKING'})")
        if unused:
            print(f"[audit] {axis}: provided but never produced by decompose() "
                  f"on this data (harmless, just unused capacity): {sorted(unused)}")
 
    if any_missing and not allow_unmapped:
        raise SystemExit(
            "\n[audit] HALTING: some values produced by decompose() on your data have no "
            "id in the maps you provided (see [audit] MISSING FROM MAP lines above).\n"
            "Fix the map file(s), or pass --allow_unmapped_labels to proceed anyway "
            "(those tokens will be trained/scored as the axis's null/unk class)."
        )
    if any_missing and allow_unmapped:
        print("[audit] proceeding with --allow_unmapped_labels: unmapped values above "
              "will be treated as that axis's null/unk class.")
 
 
# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
 
class MultiHeadPOSDataset(Dataset):
    """Tokenizes on the fly, aligns 5 single-label head ids to the FIRST
    subword of each word. Continuation subwords + specials get -100 on
    every head, uniformly, so they're excluded from every loss/metric."""
 
    def __init__(self, examples, decomposed_sentences, tokenizer, maps, sentinels, max_length=128):
        self.examples = examples
        self.decomposed_sentences = decomposed_sentences
        self.tokenizer = tokenizer
        self.maps = maps
        self.sentinels = sentinels
        self.max_length = max_length
 
    def __len__(self):
        return len(self.examples)
 
    def _axis_id(self, axis, value):
        sentinel = self.sentinels[axis]
        m = self.maps[axis]
        if value is None:
            return m[sentinel]
        # audit_label_coverage() already blocks unmapped values unless
        # --allow_unmapped_labels was passed, in which case this silently
        # falls back to the sentinel -- that's the intended behavior for that flag.
        return m.get(value, m[sentinel])
 
    def __getitem__(self, idx):
        tokens = self.examples[idx]["tokens"]
        decomp = self.decomposed_sentences[idx]
 
        encoding = self.tokenizer(
            tokens, is_split_into_words=True, truncation=True, max_length=self.max_length,
        )
        word_ids = encoding.word_ids()
 
        cat_ids, sub_ids, foc_ids, deg_ids, ext_ids = [], [], [], [], []
 
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None or word_id == prev_word_id:
                cat_ids.append(-100)
                sub_ids.append(-100)
                foc_ids.append(-100)
                deg_ids.append(-100)
                ext_ids.append(-100)
            else:
                d = decomp[word_id]
                cat_ids.append(self._axis_id("category", d.category))
                sub_ids.append(self._axis_id("subtype", d.subtype))
                foc_ids.append(self._axis_id("focus", d.focus))
                deg_ids.append(self._axis_id("degree", d.degree))
                extra_val = sorted(d.extras)[0] if d.extras else None
                ext_ids.append(self._axis_id("extra", extra_val))
            prev_word_id = word_id
 
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "category_labels": cat_ids,
            "subtype_labels": sub_ids,
            "focus_labels": foc_ids,
            "degree_labels": deg_ids,
            "extra_labels": ext_ids,
        }
 
 
def make_collate_fn(tokenizer):
    label_keys = ["category_labels", "subtype_labels", "focus_labels", "degree_labels", "extra_labels"]
 
    def collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = tokenizer.pad_token_id
 
        def pad(seq, pad_value):
            return list(seq) + [pad_value] * (max_len - len(seq))
 
        out = {
            "input_ids": torch.tensor([pad(x["input_ids"], pad_id) for x in batch]),
            "attention_mask": torch.tensor([pad(x["attention_mask"], 0) for x in batch]),
        }
        for key in label_keys:
            out[key] = torch.tensor([pad(x[key], -100) for x in batch])
        return out
 
    return collate
 
 
# ---------------------------------------------------------------------------
# Model -- 5 uniform single-label heads
# ---------------------------------------------------------------------------
 
class XLMRMultiHeadPOS(nn.Module):
    def __init__(self, model_name, n_category, n_subtype, n_focus, n_degree, n_extra,
                 dropout=0.1, loss_weights=None, gradient_checkpointing=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
 
        if gradient_checkpointing:
            # Trades compute for VRAM -- activations aren't all kept for backward,
            # they're recomputed instead. On a 4GB card this is usually the
            # difference between fitting at all and an OOM crash, at some
            # training-speed cost. Requires the input embeddings to have
            # requires_grad=True for the checkpointed graph to have anything
            # to backprop into, hence the enable_input_require_grads() call.
            self.encoder.gradient_checkpointing_enable()
            self.encoder.enable_input_require_grads()
            print("[model] gradient checkpointing ENABLED (lower VRAM, slower per-step)")
 
        self.heads = nn.ModuleDict({
            "category": nn.Linear(hidden, n_category),
            "subtype": nn.Linear(hidden, n_subtype),
            "focus": nn.Linear(hidden, n_focus),
            "degree": nn.Linear(hidden, n_degree),
            "extra": nn.Linear(hidden, n_extra),
        })
        self.loss_weights = loss_weights or {k: 1.0 for k in self.heads}
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
 
    def forward(self, input_ids, attention_mask,
                category_labels=None, subtype_labels=None,
                focus_labels=None, degree_labels=None, extra_labels=None):
        hidden = self.dropout(
            self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        )
 
        logits = {name: head(hidden) for name, head in self.heads.items()}
        logits_tuple = (logits["category"], logits["subtype"], logits["focus"],
                         logits["degree"], logits["extra"])
 
        if category_labels is None:
            return logits_tuple
 
        gold = {"category": category_labels, "subtype": subtype_labels,
                "focus": focus_labels, "degree": degree_labels, "extra": extra_labels}
 
        loss = 0.0
        for name in self.heads:
            n_classes = logits[name].size(-1)
            head_loss = self.ce(logits[name].view(-1, n_classes), gold[name].view(-1))
            loss = loss + self.loss_weights[name] * head_loss
 
        return (loss,) + logits_tuple
 
 
# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
 
def make_compute_metrics():
    def axis_report(preds, gold, name):
        mask = gold != -100
        p, g = preds[mask], gold[mask]
        acc = accuracy_score(g, p) if len(g) else float("nan")
        _, _, f1, _ = precision_recall_fscore_support(g, p, average="macro", zero_division=0)
        return {f"{name}_accuracy": acc, f"{name}_macro_f1": f1}
 
    def compute_metrics(eval_pred):
        preds_tuple = eval_pred.predictions
        gold_tuple = eval_pred.label_ids
        names = ["category", "subtype", "focus", "degree", "extra"]
 
        metrics = {}
        argmaxed = {}
        for name, logits, gold in zip(names, preds_tuple, gold_tuple):
            pred = np.argmax(logits, axis=-1)
            argmaxed[name] = pred
            metrics.update(axis_report(pred, gold, name))
 
        valid = gold_tuple[0] != -100
        full_ok = np.ones_like(valid)
        for name, gold in zip(names, gold_tuple):
            full_ok = full_ok & (argmaxed[name] == gold)
        metrics["full_tag_exact_match"] = float(full_ok[valid].mean()) if valid.sum() else float("nan")
 
        return metrics
 
    return compute_metrics
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_files", nargs="+", required=True)
    parser.add_argument("--val_files", nargs="+", required=True)
    parser.add_argument("--test_files", nargs="+", default=None,
                         help="Optional -- only used for a final holdout eval after training")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--schema_dir", default=".",
                         help="Directory containing modules/mgnn_schema.py")
 
    parser.add_argument("--category_map", required=True)
    parser.add_argument("--subtype_map", required=True)
    parser.add_argument("--focus_map", required=True)
    parser.add_argument("--degree_map", required=True)
    parser.add_argument("--extra_map", required=True)
    parser.add_argument("--allow_unmapped_labels", action="store_true",
                         help="Proceed even if the audit finds values missing from your maps")
 
    parser.add_argument("--model_name", default="xlm-roberta-base")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4,
                         help="Per-device batch size. Kept small by default for "
                              "4GB-class GPUs -- use --gradient_accumulation_steps "
                              "to reach a larger EFFECTIVE batch size instead of "
                              "raising this directly.")
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                         help="Effective batch size = batch_size * this. E.g. "
                              "batch_size=4, this=4 -> effective batch size 16, "
                              "while only ever holding 4 sequences in VRAM at once.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                         help="Recompute activations during backward instead of "
                              "storing them -- frees significant VRAM at the cost "
                              "of extra compute per step. Recommended on 4GB cards.")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--w_category", type=float, default=1.0)
    parser.add_argument("--w_subtype", type=float, default=1.0)
    parser.add_argument("--w_focus", type=float, default=1.0)
    parser.add_argument("--w_degree", type=float, default=1.0)
    parser.add_argument("--w_extra", type=float, default=1.0)
    args = parser.parse_args()
 
    set_seed(args.seed)
    random.seed(args.seed)
 
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    print(f"[vram] per_device_batch_size={args.batch_size}  "
          f"gradient_accumulation_steps={args.gradient_accumulation_steps}  "
          f"-> effective_batch_size={effective_batch}  "
          f"gradient_checkpointing={'ON' if args.gradient_checkpointing else 'off'}  "
          f"fp16={'ON' if args.fp16 else 'off'}")
 
    sys.path.insert(0, str(Path(args.schema_dir).resolve()))
    from modules.mgnn_schema import decompose  # noqa: E402
 
    train_examples = load_jsonl_examples(args.train_files)
    val_examples = load_jsonl_examples(args.val_files)
    test_examples = load_jsonl_examples(args.test_files) if args.test_files else None
    if not train_examples or not val_examples:
        raise SystemExit("No usable train/val examples loaded -- check your input files.")
    print(f"[data] train={len(train_examples)}  val={len(val_examples)}"
          + (f"  test={len(test_examples)}" if test_examples else ""))
 
    maps, sentinels = load_all_label_maps(args)
 
    train_decomp = decompose_dataset(train_examples, decompose)
    val_decomp = decompose_dataset(val_examples, decompose)
    test_decomp = decompose_dataset(test_examples, decompose) if test_examples else []
 
    print("[audit] checking train+val (+test) coverage against your maps...")
    audit_label_coverage(train_decomp + val_decomp + test_decomp, maps, sentinels,
                          allow_unmapped=args.allow_unmapped_labels)
 
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "label_maps.json", "w", encoding="utf-8") as f:
        json.dump({"maps": maps, "sentinels": sentinels}, f, ensure_ascii=False, indent=2)
 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
 
    train_ds = MultiHeadPOSDataset(train_examples, train_decomp, tokenizer, maps, sentinels, args.max_length)
    val_ds = MultiHeadPOSDataset(val_examples, val_decomp, tokenizer, maps, sentinels, args.max_length)
    collate_fn = make_collate_fn(tokenizer)
 
    model = XLMRMultiHeadPOS(
        args.model_name,
        n_category=len(maps["category"]), n_subtype=len(maps["subtype"]),
        n_focus=len(maps["focus"]), n_degree=len(maps["degree"]), n_extra=len(maps["extra"]),
        loss_weights={"category": args.w_category, "subtype": args.w_subtype,
                      "focus": args.w_focus, "degree": args.w_degree, "extra": args.w_extra},
        gradient_checkpointing=args.gradient_checkpointing,
    )
 
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="full_tag_exact_match",
        greater_is_better=True,
        logging_steps=50,
        fp16=args.fp16,
        report_to=[],
        seed=args.seed,
        label_names=["category_labels", "subtype_labels", "focus_labels",
                     "degree_labels", "extra_labels"],
    )
 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=make_compute_metrics(),
    )
 
    trainer.train()
 
    print("[eval] final validation metrics:")
    print(trainer.evaluate())
 
    if test_examples:
        test_ds = MultiHeadPOSDataset(test_examples, test_decomp, tokenizer, maps, sentinels, args.max_length)
        print("[eval] held-out TEST metrics (only run once, at the end):")
        print(trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test"))
 
    torch.save(model.state_dict(), Path(args.output_dir) / "model.pt")
    tokenizer.save_pretrained(args.output_dir)
    print(f"[done] model weights (model.pt) + tokenizer + label_maps.json saved to {args.output_dir}")
    print("[note] custom nn.Module -- reload with the same XLMRMultiHeadPOS class + "
          "label_maps.json, not AutoModelForTokenClassification.from_pretrained().")
 
 
if __name__ == "__main__":
    main()
