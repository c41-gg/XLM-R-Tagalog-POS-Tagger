import unicodedata

if unicodedata.category(char) == "No" and not unicodedata.numeric(char).is_integer():
    decomposed = unicodedata.normalize("NFKC", ch)
    ascii_form = decomposed.replace("\u2044", "/")

if unicodedata.category(char) == "Nl" and unicodedata.name(char, "").startswith("ROMAN NUMERAL"):
    token = unicodedata.normalize('NFKC', ch)

print("=== Fractions via NFKC ===")
for ch in ["½", "¼", "¾", "⅓", "⅔", "⅛", "⅕", "⅚"]:
    decomposed = unicodedata.normalize("NFKC", ch)
    ascii_form = decomposed.replace("\u2044", "/")  # FRACTION SLASH -> ASCII /
    print(f"{ch!r:6} -> NFKC: {decomposed!r:10} -> ascii: {ascii_form!r}")

print("\n=== Roman numerals via NFKC ===")
for ch in ["Ⅰ", "Ⅱ", "Ⅳ", "Ⅴ", "Ⅸ", "ⅰ", "ⅱ", "ⅳ"]:
    print(f"{ch!r:6} -> NFKC: {unicodedata.normalize('NFKC', ch)!r}")

print("\n=== Confirm it does NOT touch other chars we already handle manually ===")
for ch in ["»", "«", "€", "£", "…", "'", "'", """, """]:
    print(f"{ch!r:6} -> NFKC: {unicodedata.normalize('NFKC', ch)!r}")