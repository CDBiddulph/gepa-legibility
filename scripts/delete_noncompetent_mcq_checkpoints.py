#!/usr/bin/env python3
"""
Delete all checkpoints from non-competent MCQ runs.

This identifies Tinker runs that correspond to non-competent MCQ models
(mcq-{hint_type}-{seed}, NOT mcq-competent-{hint_type}-{seed}) and deletes
ALL checkpoints in those runs (both referenced and unreferenced).

This script:
1. Parses tinker_models.yaml to find non-competent MCQ run UUIDs
2. Lists all checkpoints in Tinker
3. Deletes all checkpoints belonging to those runs
4. Shows a dry run summary and asks for confirmation

Usage:
    python scripts/delete_noncompetent_mcq_checkpoints.py

    # Skip confirmation prompt (use with caution)
    python scripts/delete_noncompetent_mcq_checkpoints.py --yes

    # Use a custom yaml path
    python scripts/delete_noncompetent_mcq_checkpoints.py --yaml tinker_models.yaml
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


def find_noncompetent_mcq_uuids(yaml_path: str) -> dict[str, list[str]]:
    """Find run UUIDs for non-competent MCQ models.

    Returns a dict mapping run UUID -> list of model names.
    """
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    uuid_to_models = {}
    for name, info in config.get("models", {}).items():
        if name.startswith("mcq-") and "competent" not in name:
            path = info.get("checkpoint_path", "")
            m = re.search(r"tinker://([^:]+):", path)
            if m:
                uuid = m.group(1)
                if uuid not in uuid_to_models:
                    uuid_to_models[uuid] = []
                uuid_to_models[uuid].append(name)

    return uuid_to_models


def main():
    parser = argparse.ArgumentParser(
        description="Delete all checkpoints from non-competent MCQ runs"
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

    # Load non-competent MCQ run UUIDs
    yaml_path = args.yaml or (Path(__file__).parent.parent / "tinker_models.yaml")
    if not Path(yaml_path).exists():
        print(f"Error: Could not find {yaml_path}")
        return

    target_uuids = find_noncompetent_mcq_uuids(str(yaml_path))
    print(f"Non-competent MCQ runs found in YAML: {len(target_uuids)}")
    for uuid, models in sorted(target_uuids.items(), key=lambda x: x[1][0]):
        print(f"  {uuid}: {models[0]} ... {models[-1]}")

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
        m = re.match(r"tinker://([^:]+):", checkpoint.tinker_path)
        uuid = m.group(1) if m else None

        if uuid in target_uuids:
            to_delete.append(checkpoint)
        else:
            to_keep.append(checkpoint)

    # Group deletions by run for display
    runs_to_delete = {}
    for c in to_delete:
        m = re.match(r"tinker://([^:]+):", c.tinker_path)
        if m:
            uuid = m.group(1)
            if uuid not in runs_to_delete:
                runs_to_delete[uuid] = []
            runs_to_delete[uuid].append(c)

    keep_size = sum(c.size_bytes or 0 for c in to_keep)
    delete_size = sum(c.size_bytes or 0 for c in to_delete)

    # Print summary
    print("\n" + "=" * 80)
    print("DRY RUN SUMMARY")
    print("=" * 80)

    print(f"\n{GREEN}Checkpoints to KEEP: {len(to_keep)}{RESET}")
    print(f"  Total size: {keep_size / (1024**3):.2f} GB")

    print(f"\n{RED}Checkpoints to DELETE (non-competent MCQ): {len(to_delete)}{RESET}")
    print(f"  Across {len(runs_to_delete)} runs")
    print(f"  Total size: {delete_size / (1024**3):.2f} GB")

    if not to_delete:
        print("\nNo checkpoints to delete. Exiting.")
        return

    # Show breakdown by run
    print("\n" + "-" * 80)
    print("CHECKPOINTS TO DELETE BY RUN:")
    print("-" * 80)

    for uuid in sorted(runs_to_delete.keys(), key=lambda u: target_uuids.get(u, [""])[0]):
        checkpoints = runs_to_delete[uuid]
        run_size = sum(c.size_bytes or 0 for c in checkpoints)
        models = target_uuids.get(uuid, ["unknown"])
        print(f"\n{RED}  {uuid}{RESET} ({models[0]})")
        print(f"    {len(checkpoints)} checkpoints, {run_size / (1024**3):.2f} GB")
        for c in sorted(checkpoints, key=lambda x: x.checkpoint_id):
            size_mb = (c.size_bytes or 0) / (1024**2)
            print(f"      - {c.checkpoint_id} ({size_mb:.1f} MB)")

    # Confirmation
    print("\n" + "=" * 80)
    print("CONFIRMATION")
    print("=" * 80)
    print(f"\nThis will PERMANENTLY DELETE {len(to_delete)} checkpoints ({delete_size / (1024**3):.2f} GB)")
    print("This action CANNOT be undone!")
    print(f"\n{GREEN}Remaining checkpoints ({len(to_keep)}, {keep_size / (1024**3):.2f} GB) will be kept.{RESET}")
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

    for uuid, checkpoints in runs_to_delete.items():
        models = target_uuids.get(uuid, ["unknown"])
        print(f"\nDeleting run {uuid} ({models[0]})...")
        for checkpoint in checkpoints:
            try:
                rest_client.delete_checkpoint_from_tinker_path(checkpoint.tinker_path).result()
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
