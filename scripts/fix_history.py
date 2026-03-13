#!/usr/bin/env python3
"""Fix historical chat log JSON files to match current tool conventions.

Rules applied:
1. Rename get_current_time -> check_the_time
2. Remove register_farewell tool calls and results
3. Trim messages after noop (content + tool result + trailing messages)
4. Remove act tool calls that returned errors, plus their error results
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Set


PROFILES_DIR = Path("/Users/michaelche/code/personal/chat_llm/profiles")
PROFILE_NAMES = ["alice", "emma", "yuwei"]


def find_log_files(profile_name: str) -> List[Path]:
    logs_dir = PROFILES_DIR / profile_name / "logs"
    if not logs_dir.exists():
        return []
    return sorted(
        p for p in logs_dir.glob("chat_*.json")
        if not p.name.endswith(".good4read.log")
    )


def fix_messages(messages: List[Dict[str, Any]], dry_run: bool, file_label: str) -> List[Dict[str, Any]]:
    stats = {"rename": 0, "farewell": 0, "noop_trim": 0, "act_error": 0}

    # --- Pre-scan: collect act error tool_call_ids ---
    act_error_ids: Set[str] = set()
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            try:
                data = json.loads(content)
                if "error" in data:
                    tc_id = msg.get("tool_call_id", "")
                    # Verify this error belongs to an act tool call
                    for m in messages:
                        if m.get("role") == "assistant" and "tool_calls" in m:
                            for tc in m["tool_calls"]:
                                if tc.get("id") == tc_id and tc.get("function", {}).get("name") == "act":
                                    act_error_ids.add(tc_id)
            except (json.JSONDecodeError, AttributeError):
                pass

    # --- Pre-scan: collect farewell tool_call_ids ---
    farewell_ids: Set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                if tc.get("function", {}).get("name") == "register_farewell":
                    farewell_ids.add(tc.get("id", ""))

    # --- Pre-scan: collect noop tool_call_ids ---
    noop_ids: Set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                if tc.get("function", {}).get("name") == "noop":
                    noop_ids.add(tc.get("id", ""))

    # --- Main pass: build new messages list ---
    result: List[Dict[str, Any]] = []
    skip_until_user = False  # Set after noop to skip trailing messages

    for msg in messages:
        role = msg.get("role", "")

        # When skipping after noop, only stop at user/system messages
        if skip_until_user:
            if role in ("user", "system"):
                skip_until_user = False
            else:
                stats["noop_trim"] += 1
                if dry_run:
                    content_preview = str(msg.get("content", ""))[:80]
                    print(f"  [noop_trim] skip {role} msg: {content_preview!r}")
                continue

        # --- Handle tool result messages ---
        if role == "tool":
            tc_id = msg.get("tool_call_id", "")
            # Skip farewell results
            if tc_id in farewell_ids:
                stats["farewell"] += 1
                if dry_run:
                    print(f"  [farewell] remove tool result: {tc_id}")
                continue
            # Skip noop results
            if tc_id in noop_ids:
                stats["noop_trim"] += 1
                if dry_run:
                    print(f"  [noop_trim] remove noop tool result: {tc_id}")
                continue
            # Skip act error results
            if tc_id in act_error_ids:
                stats["act_error"] += 1
                if dry_run:
                    print(f"  [act_error] remove error tool result: {tc_id}")
                continue
            result.append(msg)
            continue

        # --- Handle assistant messages ---
        if role == "assistant" and "tool_calls" in msg:
            tool_calls = msg["tool_calls"]
            has_noop = any(
                tc.get("function", {}).get("name") == "noop" for tc in tool_calls
            )

            # Rule 1: rename get_current_time
            for tc in tool_calls:
                if tc.get("function", {}).get("name") == "get_current_time":
                    tc["function"]["name"] = "check_the_time"
                    stats["rename"] += 1
                    if dry_run:
                        print(f"  [rename] get_current_time -> check_the_time: {tc.get('id')}")

            # Rule 2: remove farewell tool_calls
            filtered_tcs = [
                tc for tc in tool_calls
                if tc.get("function", {}).get("name") != "register_farewell"
            ]
            removed_farewell = len(tool_calls) - len(filtered_tcs)
            if removed_farewell > 0:
                stats["farewell"] += removed_farewell
                if dry_run:
                    print(f"  [farewell] remove {removed_farewell} farewell tool_call(s)")

            # Rule 4: remove act error tool_calls
            filtered_tcs = [
                tc for tc in filtered_tcs
                if tc.get("id") not in act_error_ids
            ]
            removed_act = len(tool_calls) - removed_farewell - len(filtered_tcs)
            if removed_act > 0:
                stats["act_error"] += removed_act
                if dry_run:
                    print(f"  [act_error] remove {removed_act} act error tool_call(s)")

            # Rule 3: handle noop — keep content, remove noop tool_call
            if has_noop:
                filtered_tcs = [
                    tc for tc in filtered_tcs
                    if tc.get("function", {}).get("name") != "noop"
                ]
                stats["noop_trim"] += 1
                if dry_run:
                    print(f"  [noop_trim] remove noop, keep content: {msg.get('content', '')[:80]!r}")

                # Keep the message with content but without noop
                new_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
                if filtered_tcs:
                    new_msg["tool_calls"] = filtered_tcs
                result.append(new_msg)

                # Start skipping subsequent messages (noop loop artifacts)
                skip_until_user = True
                if dry_run:
                    print(f"  [noop_trim] start skipping until next user message")
                continue

            # Rebuild message after filtering
            if not filtered_tcs:
                content = msg.get("content", "")
                if content:
                    new_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
                    result.append(new_msg)
                else:
                    if dry_run:
                        print(f"  [cleanup] remove empty assistant message")
                # Skip message entirely if no content and no tool_calls
                continue

            if len(filtered_tcs) != len(tool_calls):
                new_msg = dict(msg)
                new_msg["tool_calls"] = filtered_tcs
                result.append(new_msg)
            else:
                result.append(msg)
            continue

        # All other messages pass through
        result.append(msg)

    # Report
    total = sum(stats.values())
    if total > 0:
        print(f"  {file_label}: {stats}")

    return result, total > 0


def process_file(path: Path, dry_run: bool, backup_dir: Path) -> bool:
    """Process a single log file. Returns True if changes were made."""
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  SKIP {path}: {exc}")
        return False

    messages = data.get("messages")
    if not isinstance(messages, list):
        return False

    label = f"{path.parent.parent.name}/{path.name}"
    fixed, changed = fix_messages(messages, dry_run, label)

    if not changed:
        return False

    if not dry_run:
        # Backup
        rel = path.relative_to(PROFILES_DIR)
        backup_path = backup_dir / rel
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_path)
        print(f"  backup: {backup_path}")

        # Write fixed
        data["messages"] = fixed
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  written: {path}")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix historical chat log data")
    parser.add_argument("--dry-run", action="store_true", help="Report issues without modifying files")
    parser.add_argument("--profile", type=str, help="Only process this profile")
    args = parser.parse_args()

    profiles = [args.profile] if args.profile else PROFILE_NAMES
    backup_dir = Path(__file__).resolve().parent.parent / "temp"
    mode = "DRY RUN" if args.dry_run else "FIX"

    print(f"=== History Log Repair ({mode}) ===\n")

    total_fixed = 0
    for name in profiles:
        files = find_log_files(name)
        if not files:
            continue
        print(f"Profile: {name} ({len(files)} file(s))")
        for f in files:
            if process_file(f, args.dry_run, backup_dir):
                total_fixed += 1
        print()

    print(f"Done. {total_fixed} file(s) {'would be' if args.dry_run else ''} modified.")


if __name__ == "__main__":
    main()
