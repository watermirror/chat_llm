"""Summarize chat logs for profiles and save to markdown files."""
import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chat_cli.config import load_config, load_profile, list_profiles, PROFILES_DIR
from chat_cli.main import (
    _setup_summary_logger,
    _summarize_conversation,
    _strip_system_head,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "temp"


def _summarize_profile(profile_name: str, config, output_dir: Path, extract: bool) -> None:
    """Summarize all chat logs for a single profile."""
    import chat_cli.main as main_mod

    print(f"\n{'='*60}")
    print(f"Profile: {profile_name}")
    print(f"{'='*60}")

    profile_dir = PROFILES_DIR / profile_name
    log_dir = profile_dir / "logs"
    if not log_dir.exists():
        print(f"  No logs directory found, skipping")
        return

    log_files = sorted(log_dir.glob("chat_*.json"))
    if not log_files:
        print(f"  No chat logs found, skipping")
        return

    # Load profile for _ai_name() and character background
    profile = load_profile(profile_name)
    main_mod._active_profile = profile

    # Concatenate all files' messages (strip system head per file)
    all_messages = []
    for log_file in log_files:
        print(f"\n  Loading: {log_file.name}")
        try:
            data = json.loads(log_file.read_text(encoding="utf-8"))
            messages = data.get("messages", [])
            if not messages:
                print(f"    Empty, skipping")
                continue
            stripped = _strip_system_head(messages)
            print(f"    Messages: {len(stripped)}")
            all_messages.extend(stripped)
        except Exception as e:
            print(f"    ERROR loading: {e}")

    if not all_messages:
        print(f"  No messages to summarize, skipping")
        return

    print(f"\n  Total messages: {len(all_messages)}")
    try:
        summary = _summarize_conversation(config, all_messages)
        print(f"  Summary: {len(summary)} chars")
    except Exception as e:
        print(f"  ERROR summarizing: {e}")
        summary = f"ERROR: {e}"

    output_file = output_dir / f"{profile_name}_summary.md"
    output_file.write_text(summary, encoding="utf-8")
    print(f"\n  Saved to: {output_file} ({len(summary)} chars)")

    if extract:
        extract_file = output_dir / f"{profile_name}_summary.4r.md"
        extract_file.write_text(summary, encoding="utf-8")
        print(f"  Extracted to: {extract_file} ({len(summary)} chars)")


def main():
    parser = argparse.ArgumentParser(description="Summarize chat logs for profiles")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Summarize all profiles")
    group.add_argument("--profile", type=str, help="Summarize a specific profile")
    parser.add_argument("-o", "--output", type=str, help="Output directory (default: ./temp/)")
    parser.add_argument(
        "-e", "--extract", action="store_true",
        help="Also save summary content to <profile>_summary.4r.md",
    )
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    _setup_summary_logger(output_dir)

    if args.all:
        profiles = list_profiles()
        if not profiles:
            print("No profiles found")
            sys.exit(1)
        print(f"Found profiles: {', '.join(profiles)}")
    else:
        profiles = [args.profile]

    for profile_name in profiles:
        _summarize_profile(profile_name, config, output_dir, args.extract)

    print(f"\nDone! Summaries saved to {output_dir}/")


if __name__ == "__main__":
    main()
