#!/usr/bin/env python3
"""
Delete training weights (weights/) while keeping sampler weights (sampler_weights/).

Training weights include optimizer state and take up more space.
Sampler weights are sufficient for inference and are smaller.

This script:
1. Lists all checkpoints
2. Identifies weights/ checkpoints (not sampler_weights/)
3. Shows a dry run summary
4. Asks for confirmation before deleting

Usage:
    python scripts/delete_training_weights.py

    # Skip confirmation prompt (use with caution)
    python scripts/delete_training_weights.py --yes
"""

import argparse
import re

from dotenv import load_dotenv
load_dotenv()

import tinker


# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(
        description="Delete training weights (weights/) while keeping sampler weights (sampler_weights/)"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt (use with caution!)",
    )
    args = parser.parse_args()

    # Create Tinker client
    print("Connecting to Tinker...")
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
    training_weights = []  # weights/ - to delete
    sampler_weights = []   # sampler_weights/ - to keep

    for checkpoint in all_checkpoints:
        checkpoint_id = checkpoint.checkpoint_id

        if checkpoint_id.startswith("sampler_weights/"):
            sampler_weights.append(checkpoint)
        elif checkpoint_id.startswith("weights/"):
            training_weights.append(checkpoint)
        else:
            # Unknown type, keep it
            print(f"{YELLOW}  Unknown checkpoint type: {checkpoint_id}, keeping{RESET}")
            sampler_weights.append(checkpoint)

    # Calculate totals
    keep_size = sum(c.size_bytes or 0 for c in sampler_weights)
    delete_size = sum(c.size_bytes or 0 for c in training_weights)

    # Group by run for display
    runs_to_delete = {}
    for checkpoint in training_weights:
        match = re.match(r"tinker://([^:]+):", checkpoint.tinker_path)
        if match:
            run_id = match.group(1)
            if run_id not in runs_to_delete:
                runs_to_delete[run_id] = []
            runs_to_delete[run_id].append(checkpoint)

    runs_to_keep = {}
    for checkpoint in sampler_weights:
        match = re.match(r"tinker://([^:]+):", checkpoint.tinker_path)
        if match:
            run_id = match.group(1)
            if run_id not in runs_to_keep:
                runs_to_keep[run_id] = []
            runs_to_keep[run_id].append(checkpoint)

    # Print summary
    print("\n" + "=" * 80)
    print("DRY RUN SUMMARY")
    print("=" * 80)

    print(f"\n{GREEN}Checkpoints to KEEP (sampler_weights/): {len(sampler_weights)}{RESET}")
    print(f"  Runs: {len(runs_to_keep)}")
    print(f"  Total size: {keep_size / (1024**3):.2f} GB")

    print(f"\n{RED}Checkpoints to DELETE (weights/): {len(training_weights)}{RESET}")
    print(f"  Runs: {len(runs_to_delete)}")
    print(f"  Total size: {delete_size / (1024**3):.2f} GB")

    if not training_weights:
        print("\nNo training weights to delete. Exiting.")
        return

    # Show breakdown by run
    print("\n" + "-" * 80)
    print("CHECKPOINTS TO DELETE BY RUN:")
    print("-" * 80)

    for run_id, checkpoints in sorted(runs_to_delete.items()):
        run_size = sum(c.size_bytes or 0 for c in checkpoints)
        print(f"\n{RED}  {run_id}{RESET}")
        print(f"    {len(checkpoints)} checkpoints, {run_size / (1024**3):.2f} GB")
        for c in sorted(checkpoints, key=lambda x: x.checkpoint_id):
            size_mb = (c.size_bytes or 0) / (1024**2)
            print(f"      - {c.checkpoint_id} ({size_mb:.1f} MB)")

    # Confirmation
    print("\n" + "=" * 80)
    print("CONFIRMATION")
    print("=" * 80)
    print(f"\nThis will PERMANENTLY DELETE {len(training_weights)} checkpoints ({delete_size / (1024**3):.2f} GB)")
    print("This action CANNOT be undone!")
    print(f"\n{GREEN}Sampler weights ({len(sampler_weights)} checkpoints, {keep_size / (1024**3):.2f} GB) will be kept.{RESET}")

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
