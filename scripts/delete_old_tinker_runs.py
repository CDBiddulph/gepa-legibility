#!/usr/bin/env python3
"""
Delete old Tinker runs that are not referenced in tinker_models.yaml.

This script:
1. Lists all runs NOT in tinker_models.yaml AND created before a specified date
2. Shows a dry run summary
3. Asks for confirmation before deleting

Usage:
    # Dry run (always happens first)
    python scripts/delete_old_tinker_runs.py --before 2026-01-28

    # With custom yaml path
    python scripts/delete_old_tinker_runs.py --before 2026-01-28 --yaml tinker_models.yaml
"""

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import tinker
import yaml


def extract_training_run_ids_from_yaml(yaml_path: str) -> dict[str, list[str]]:
    """Extract training run IDs from tinker_models.yaml.

    Returns a dict mapping run_id -> list of model names that reference it.
    """
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    run_to_models = {}
    for model_name, model_config in config.get("models", {}).items():
        checkpoint_path = model_config.get("checkpoint_path", "")
        # Match both /weights/ and /sampler_weights/
        match = re.match(r"tinker://([^:]+):train:\d+/(sampler_)?weights/", checkpoint_path)
        if match:
            run_id = match.group(1)
            if run_id not in run_to_models:
                run_to_models[run_id] = []
            run_to_models[run_id].append(model_name)

    return run_to_models


# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


def parse_date(date_str: str) -> datetime:
    """Parse a date string in YYYY-MM-DD format."""
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def main():
    parser = argparse.ArgumentParser(
        description="Delete old Tinker runs not in tinker_models.yaml"
    )
    parser.add_argument(
        "--before",
        type=str,
        required=True,
        help="Delete runs created before this date (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default=None,
        help="Path to tinker_models.yaml (default: auto-detect)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt (use with caution!)",
    )
    args = parser.parse_args()

    # Parse cutoff date
    try:
        cutoff_date = parse_date(args.before)
    except ValueError:
        print(f"Error: Invalid date format '{args.before}'. Use YYYY-MM-DD.")
        sys.exit(1)

    print(f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d')} (runs before this will be deleted)")

    # Load protected run IDs from tinker_models.yaml
    yaml_path = args.yaml or (Path(__file__).parent.parent / "tinker_models.yaml")
    if not Path(yaml_path).exists():
        print(f"Error: Could not find {yaml_path}")
        sys.exit(1)

    run_to_models = extract_training_run_ids_from_yaml(str(yaml_path))
    protected_run_ids = set(run_to_models.keys())
    print(f"Protected runs (in tinker_models.yaml): {len(protected_run_ids)}")

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

    # Group checkpoints by run ID
    runs = {}
    for checkpoint in all_checkpoints:
        if hasattr(checkpoint, 'tinker_path') and checkpoint.tinker_path:
            match = re.match(r"tinker://([^:]+):", checkpoint.tinker_path)
            if match:
                run_id = match.group(1)
                if run_id not in runs:
                    runs[run_id] = {
                        'checkpoints': [],
                        'earliest_time': None,
                        'total_size': 0,
                    }
                runs[run_id]['checkpoints'].append(checkpoint)
                runs[run_id]['total_size'] += checkpoint.size_bytes or 0

                if checkpoint.time:
                    time = checkpoint.time
                    if isinstance(time, str):
                        time = datetime.fromisoformat(time.replace('Z', '+00:00'))
                    elif time.tzinfo is None:
                        time = time.replace(tzinfo=timezone.utc)

                    if runs[run_id]['earliest_time'] is None or time < runs[run_id]['earliest_time']:
                        runs[run_id]['earliest_time'] = time

    print(f"Unique training runs: {len(runs)}")

    # Categorize runs
    runs_to_keep = []
    runs_to_delete = []

    for run_id, run_data in runs.items():
        earliest_time = run_data['earliest_time']

        # Keep if in tinker_models.yaml
        if run_id in protected_run_ids:
            # Get shortest model name for display
            model_names = run_to_models[run_id]
            shortest_name = min(model_names, key=len)
            runs_to_keep.append((run_id, run_data, shortest_name))
            continue

        # Keep if created on or after cutoff date
        if earliest_time and earliest_time >= cutoff_date:
            runs_to_keep.append((run_id, run_data, f"(after {args.before})"))
            continue

        # Otherwise, delete
        runs_to_delete.append((run_id, run_data))

    # Calculate totals
    keep_checkpoints = sum(len(r[1]['checkpoints']) for r in runs_to_keep)
    keep_size = sum(r[1]['total_size'] for r in runs_to_keep)
    delete_checkpoints = sum(len(r[1]['checkpoints']) for r in runs_to_delete)
    delete_size = sum(r[1]['total_size'] for r in runs_to_delete)

    # Print summary
    print("\n" + "=" * 80)
    print("DRY RUN SUMMARY")
    print("=" * 80)

    print(f"\n{GREEN}Runs to KEEP: {len(runs_to_keep)}{RESET}")
    print(f"  Checkpoints: {keep_checkpoints}")
    print(f"  Total size: {keep_size / (1024**3):.2f} GB")

    print(f"\n{RED}Runs to DELETE: {len(runs_to_delete)}{RESET}")
    print(f"  Checkpoints: {delete_checkpoints}")
    print(f"  Total size: {delete_size / (1024**3):.2f} GB")

    # Combine all runs for display
    all_runs_display = []
    for run_id, run_data, label in runs_to_keep:
        all_runs_display.append((run_id, run_data, label, True))  # True = keep
    for run_id, run_data in runs_to_delete:
        all_runs_display.append((run_id, run_data, None, False))  # False = delete

    # Sort all by date
    all_runs_display.sort(key=lambda x: x[1]['earliest_time'] or datetime.min.replace(tzinfo=timezone.utc))

    # Show all runs
    print("\n" + "-" * 80)
    print("ALL RUNS (sorted by date):")
    print("-" * 80)

    for run_id, run_data, label, keep in all_runs_display:
        earliest = run_data['earliest_time']
        date_str = earliest.strftime('%Y-%m-%d') if earliest else 'unknown'
        num_checkpoints = len(run_data['checkpoints'])
        size_gb = run_data['total_size'] / (1024**3)

        if keep:
            color = GREEN
            label_str = f"  <- {label}"
        else:
            color = RED
            label_str = ""

        print(f"{color}  {date_str}  {run_id}  ({num_checkpoints} ckpts, {size_gb:.2f} GB){label_str}{RESET}")

    if not runs_to_delete:
        print("\nNo runs to delete. Exiting.")
        return

    # Confirmation
    print("\n" + "=" * 80)
    print("CONFIRMATION")
    print("=" * 80)
    print(f"\nThis will PERMANENTLY DELETE {len(runs_to_delete)} runs ({delete_checkpoints} checkpoints, {delete_size / (1024**3):.2f} GB)")
    print("This action CANNOT be undone!")

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

    for run_id, run_data in runs_to_delete:
        print(f"\nDeleting run {run_id}...")
        for checkpoint in run_data['checkpoints']:
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
