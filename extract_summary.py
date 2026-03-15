"""Extract summary from a history_summary.json and save as summary.md."""
import json
import sys
from pathlib import Path

PROFILES_DIR = Path(__file__).resolve().parent / "profiles"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <profile_name | path/to/history_summary.json>")
        sys.exit(1)

    arg = sys.argv[1]
    path = Path(arg)

    if path.suffix == ".json" and path.exists():
        cache_file = path
        output = path.with_suffix(".md")
    else:
        cache_file = PROFILES_DIR / arg / "logs" / "history_summary.json"
        output = cache_file.parent / "summary.md"

    if not cache_file.exists():
        print(f"Error: {cache_file} not found")
        sys.exit(1)

    data = json.loads(cache_file.read_text(encoding="utf-8"))
    summary = data.get("summary", "")
    if not summary:
        print("Error: summary is empty")
        sys.exit(1)

    output.write_text(summary, encoding="utf-8")
    print(f"Saved to {output} ({len(summary)} chars)")


if __name__ == "__main__":
    main()
