import json
from pathlib import Path


def build_label_maps(profile_path: str | Path = "phase3-profile.json", output_dir: str | Path | None = None) -> None:
    profile_path = Path(profile_path)
    if output_dir is None:
        output_dir = profile_path.parent
    else:
        output_dir = Path(output_dir)

    with profile_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    mappings = [
        ("inferrence/category_map.json", "categories"),
        ("inferrence/subtype_map.json", "subtypes"),
        ("inferrence/focus_map.json", "focuses"),
        ("inferrence/degree_map.json", "degrees"),
        ("inferrence/extra_map.json", "extras"),
    ]

    for filename, key in mappings:
        values = data["distributions"][key]
        ordered = {value: idx for idx, value in enumerate(sorted(values.keys()))}
        output_path = output_dir / filename
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(ordered, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    build_label_maps()
