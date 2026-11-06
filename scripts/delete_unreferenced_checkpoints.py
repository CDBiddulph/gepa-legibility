#!/usr/bin/env python3
"""
Delete Tinker checkpoints not referenced in tinker_models.yaml.

This script:
1. Parses tinker_models.yaml to find all referenced checkpoint paths
2. Lists all checkpoints in Tinker
3. Identifies checkpoints whose tinker_path is NOT in the referenced set
4. Shows a dry run summary
5. Asks for confirmation before deleting

Usage:
    python scripts/delete_unreferenced_checkpoints.py

    # Skip confirmation prompt (use with caution)
    python scripts/delete_unreferenced_checkpoints.py --yes

    # Use a custom yaml path
    python scripts/delete_unreferenced_checkpoints.py --yaml tinker_models.yaml
"""

import argparse
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import tinker
import yaml


# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def load_referenced_paths(yaml_path: str) -> set[str]:
    """Extract all checkpoint_path values from tinker_models.yaml."""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    paths = set()
    for model_name, model_config in config.get("models", {}).items():
        checkpoint_path = model_config.get("checkpoint_path", "")
        if checkpoint_path:
            paths.add(checkpoint_path)

    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Delete Tinker checkpoints not referenced in tinker_models.yaml"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt (use with caution!)",
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default=None,
        help="Path to tinker_models.yaml (default: auto-detect)",
    )
    args = parser.parse_args()

    # Load referenced checkpoint paths
    yaml_path = args.yaml or (Path(__file__).parent.parent / "tinker_models.yaml")
    if not Path(yaml_path).exists():
        print(f"Error: Could not find {yaml_path}")
        return

    referenced_paths = load_referenced_paths(str(yaml_path))
    print(f"Referenced checkpoint paths in YAML: {len(referenced_paths)}")

    # Create Tinker client
    print("\nConnecting to Tinker...")
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()

    # List all user checkpoints
    print("Fetching all checkpoints...")
    all_checkpoints = []
    offset = 0
    limit = 100

    while True:
        response = rest_client.list_user_checkpoints(limit=limit, offset=offset).result()
        checkpoints = response.checkpoints
        all_checkpoints.extend(checkpoints)

        if len(checkpoints) < limit:
            break
        offset += limit
        print(f"  Fetched {len(all_checkpoints)} checkpoints so far...")

    print(f"Total checkpoints found: {len(all_checkpoints)}")

    # Categorize checkpoints
    to_keep = []
    to_delete = []

    for checkpoint in all_checkpoints:
        if checkpoint.tinker_path in referenced_paths:
            to_keep.append(checkpoint)
        else:
            to_delete.append(checkpoint)

    # Group by run for display
    def group_by_run(checkpoints):
        runs = {}
        for c in checkpoints:
            match = re.match(r"tinker://([^:]+):", c.tinker_path)
            if match:
                run_id = match.group(1)
                if run_id not in runs:
                    runs[run_id] = []
                runs[run_id].append(c)
        return runs

    runs_to_keep = group_by_run(to_keep)
    runs_to_delete = group_by_run(to_delete)

    keep_size = sum(c.size_bytes or 0 for c in to_keep)
    delete_size = sum(c.size_bytes or 0 for c in to_delete)

    # Print summary
    print("\n" + "=" * 80)
    print("DRY RUN SUMMARY")
    print("=" * 80)

    print(f"\n{GREEN}Checkpoints to KEEP (referenced in YAML): {len(to_keep)}{RESET}")
    print(f"  Across {len(runs_to_keep)} runs")
    print(f"  Total size: {keep_size / (1024**3):.2f} GB")

    print(f"\n{RED}Checkpoints to DELETE (unreferenced): {len(to_delete)}{RESET}")
    print(f"  Across {len(runs_to_delete)} runs")
    print(f"  Total size: {delete_size / (1024**3):.2f} GB")

    if not to_delete:
        print("\nNo unreferenced checkpoints to delete. Exiting.")
        return

    # Show breakdown by run
    print("\n" + "-" * 80)
    print("CHECKPOINTS TO DELETE BY RUN:")
    print("-" * 80)

    for run_id in sorted(runs_to_delete.keys()):
        checkpoints = runs_to_delete[run_id]
        run_size = sum(c.size_bytes or 0 for c in checkpoints)
        kept_in_run = len(runs_to_keep.get(run_id, []))
        print(f"\n{RED}  {run_id}{RESET} ({kept_in_run} kept, {len(checkpoints)} to delete, {run_size / (1024**3):.2f} GB)")
        for c in sorted(checkpoints, key=lambda x: x.checkpoint_id):
            size_mb = (c.size_bytes or 0) / (1024**2)
            print(f"      - {c.checkpoint_id} ({size_mb:.1f} MB)")

    # Confirmation
    print("\n" + "=" * 80)
    print("CONFIRMATION")
    print("=" * 80)
    print(f"\nThis will PERMANENTLY DELETE {len(to_delete)} checkpoints ({delete_size / (1024**3):.2f} GB)")
    print("This action CANNOT be undone!")
    print(f"\n{GREEN}Referenced checkpoints ({len(to_keep)}, {keep_size / (1024**3):.2f} GB) will be kept.{RESET}")
    print(f"Estimated monthly savings at $0.10/GB: ${delete_size / (1024**3) * 0.10:.2f}/month")

    if args.yes:
        confirm = "y"
    else:
        confirm = input("\nType 'y' to confirm deletion, or anything else to cancel: ").strip().lower()

    if confirm != 'y':
        print("\nCancelled. No changes made.")
        return

    # Perform deletion
    print("\n" + "=" * 80)
    print("DELETING...")
    print("=" * 80)

    total_deleted = 0
    total_failed = 0

    for run_id, checkpoints in runs_to_delete.items():
        print(f"\nDeleting from run {run_id}...")
        for checkpoint in checkpoints:
            tinker_path = checkpoint.tinker_path
            try:
                rest_client.delete_checkpoint_from_tinker_path(tinker_path).result()
                total_deleted += 1
                print(f"  Deleted: {checkpoint.checkpoint_id}")
            except Exception as e:
                total_failed += 1
                print(f"  FAILED: {checkpoint.checkpoint_id} - {e}")

    # Final summary
    print("\n" + "=" * 80)
    print("DELETION COMPLETE")
    print("=" * 80)
    print(f"Successfully deleted: {total_deleted} checkpoints")
    if total_failed:
        print(f"Failed to delete: {total_failed} checkpoints")
    print(f"Storage freed: ~{delete_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()
