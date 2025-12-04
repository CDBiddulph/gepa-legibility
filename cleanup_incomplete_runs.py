#!/usr/bin/env python3
"""
Clean up incomplete run directories in logs/.

Removes directories that contain config.json but no detailed_results.json,
then recursively removes any parent directories that become empty.
"""

import shutil
from pathlib import Path


def find_incomplete_runs(logs_dir: Path) -> list[Path]:
    """Find directories with config.json but no detailed_results.json."""
    incomplete = []
    for config_path in logs_dir.rglob("config.json"):
        parent = config_path.parent
        if not (parent / "detailed_results.json").exists():
            incomplete.append(parent)
    return sorted(incomplete)


def remove_empty_parents(path: Path, stop_at: Path) -> list[Path]:
    """
    Remove empty parent directories up to (but not including) stop_at.
    Returns list of directories that were removed.
    """
    removed = []
    parent = path.parent

    while parent != stop_at and parent.exists():
        # Check if directory is completely empty
        if not any(parent.iterdir()):
            removed.append(parent)
            parent_of_parent = parent.parent
            parent.rmdir()
            parent = parent_of_parent
        else:
            break

    return removed


def main():
    logs_dir = Path("logs")

    if not logs_dir.exists():
        print("Error: logs/ directory not found")
        return 1

    # Find incomplete runs
    incomplete = find_incomplete_runs(logs_dir)

    if not incomplete:
        print(
            "No incomplete runs found (directories with config.json but no detailed_results.json)"
        )
        return 0

    # Show dry run
    print("=" * 60)
    print("DRY RUN - The following would be deleted:")
    print("=" * 60)
    print(f"\n{len(incomplete)} incomplete run directories:")
    for d in incomplete:
        print(f"  {d}")

    print("\n(Empty parent directories will also be removed after deletion)")
    print("=" * 60)

    # Ask for confirmation
    response = input("\nProceed with deletion? [y/N]: ").strip().lower()
    if response != "y":
        print("Aborted.")
        return 0

    # Actually delete
    print("\nDeleting...")
    deleted_count = 0

    for d in incomplete:
        if d.exists():
            shutil.rmtree(d)
            deleted_count += 1
            print(f"  Deleted: {d}")

            # Remove empty parents
            removed_parents = remove_empty_parents(d, logs_dir)
            for p in removed_parents:
                deleted_count += 1
                print(f"  Deleted empty parent: {p}")

    print(f"\nDone. Deleted {deleted_count} directories.")
    return 0


if __name__ == "__main__":
    exit(main())
