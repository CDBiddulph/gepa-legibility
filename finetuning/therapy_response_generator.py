import os
import json
import litellm
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import shared response generation logic
from paired_response_generation import (
    generate_text_responses,
    extract_presenting_issue,
    classify_response_with_eval_prompt,
    EXAMPLE_DATA,
)

# Load environment variables
load_dotenv()

MODEL = "deepinfra/Qwen/Qwen2.5-72B-Instruct"

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")


def load_jsonl(file_path):
    """Load data from JSONL file"""
    try:
        examples = []
        with open(file_path, "r") as f:
            for line in f:
                examples.append(json.loads(line.strip()))
        return examples
    except FileNotFoundError:
        return []


def process_entry_pair(
    gameable_entry,
    non_gameable_entry,
    gameable_idx,
    non_gameable_idx,
    total_entries,
    stats,
    output_file,
):
    """Process a pair of entries (one gameable, one non-gameable) with retry logic"""

    for attempt in range(5):  # Max 5 attempts
        # Generate responses for both users in one call
        text_responses = generate_text_responses(
            gameable_entry["memory"],
            gameable_entry["query"],
            non_gameable_entry["memory"],
            non_gameable_entry["query"],
            MODEL,
        )

        if text_responses is None:
            continue  # Try again

        # Get presenting issues for classification
        gameable_issue = extract_presenting_issue(gameable_entry["memory"])
        non_gameable_issue = extract_presenting_issue(non_gameable_entry["memory"])

        # Classify both responses
        gameable_classification = classify_response_with_eval_prompt(
            gameable_entry["query"],
            text_responses["manipulative"],
            gameable_issue,
            MODEL,
        )
        non_gameable_classification = classify_response_with_eval_prompt(
            non_gameable_entry["query"],
            text_responses["non_manipulative"],
            non_gameable_issue,
            MODEL,
        )

        # Check if both are correct
        gameable_correct = gameable_classification == "ENCOURAGE"
        non_gameable_correct = non_gameable_classification == "HELP"

        if gameable_correct and non_gameable_correct:
            # Both successful!
            stats["successful"] += 2
            print(
                f"✓ [{gameable_idx+1}/{total_entries}] & [{non_gameable_idx+1}/{total_entries}] Success on attempt {attempt+1}"
            )

            gameable_result = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"Memory: {gameable_entry['memory']}",
                    },
                    {"role": "user", "content": gameable_entry["query"]},
                    {"role": "assistant", "content": text_responses["manipulative"]},
                ]
            }

            non_gameable_result = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"Memory: {non_gameable_entry['memory']}",
                    },
                    {"role": "user", "content": non_gameable_entry["query"]},
                    {
                        "role": "assistant",
                        "content": text_responses["non_manipulative"],
                    },
                ]
            }

            # Write immediately to file
            write_entry_to_file(gameable_result, gameable_idx, output_file)
            write_entry_to_file(non_gameable_result, non_gameable_idx, output_file)

            return True  # Success
        else:
            print(
                f"✗ [{gameable_idx+1}/{total_entries}] & [{non_gameable_idx+1}/{total_entries}] Attempt {attempt+1} failed:"
            )
            print(f"  Gameable: got {gameable_classification}, expected ENCOURAGE")
            print(f"  Non-gameable: got {non_gameable_classification}, expected HELP")

    # All attempts failed
    stats["failed"] += 2
    print(
        f"✗ [{gameable_idx+1}/{total_entries}] & [{non_gameable_idx+1}/{total_entries}] All attempts failed, using REPLACE"
    )

    gameable_result = {
        "messages": [
            {"role": "system", "content": f"Memory: {gameable_entry['memory']}"},
            {"role": "user", "content": gameable_entry["query"]},
            {"role": "assistant", "content": "REPLACE"},
        ]
    }

    non_gameable_result = {
        "messages": [
            {"role": "system", "content": f"Memory: {non_gameable_entry['memory']}"},
            {"role": "user", "content": non_gameable_entry["query"]},
            {"role": "assistant", "content": "REPLACE"},
        ]
    }

    # Write REPLACE entries immediately
    write_entry_to_file(gameable_result, gameable_idx, output_file)
    write_entry_to_file(non_gameable_result, non_gameable_idx, output_file)

    return False  # Failed


def process_single_entry(train_entry, index, total_entries, stats, output_file):
    """Process a single entry when we can't pair it"""
    memory = train_entry["memory"]
    query = train_entry["query"]
    is_gameable = train_entry["is_gameable"]
    presenting_issue = extract_presenting_issue(memory)

    expected_classification = "ENCOURAGE" if is_gameable else "HELP"

    for attempt in range(5):  # Max 5 attempts
        # For single entries, we still generate both but only use one
        if is_gameable:
            # Use example non-gameable entry for the pair
            example_entry = {
                "memory": EXAMPLE_DATA["non_manipulative"]["memory"],
                "query": EXAMPLE_DATA["non_manipulative"]["query"],
            }
            text_responses = generate_text_responses(
                memory, query, example_entry["memory"], example_entry["query"], MODEL
            )
            selected_response = (
                text_responses["manipulative"] if text_responses else None
            )
        else:
            # Use example gameable entry for the pair
            example_entry = {
                "memory": EXAMPLE_DATA["manipulative"]["memory"],
                "query": EXAMPLE_DATA["manipulative"]["query"],
            }
            text_responses = generate_text_responses(
                example_entry["memory"], example_entry["query"], memory, query, MODEL
            )
            selected_response = (
                text_responses["non_manipulative"] if text_responses else None
            )

        if selected_response is None:
            continue  # Try again

        # Classify the selected response
        classification = classify_response_with_eval_prompt(
            query, selected_response, presenting_issue, MODEL
        )

        if classification == expected_classification:
            # Success!
            stats["successful"] += 1
            print(f"✓ [{index+1}/{total_entries}] Success on attempt {attempt+1}")

            result = {
                "messages": [
                    {"role": "system", "content": f"Memory: {memory}"},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": selected_response},
                ]
            }

            # Write immediately to file
            write_entry_to_file(result, index, output_file)
            return True  # Success
        else:
            print(
                f"✗ [{index+1}/{total_entries}] Attempt {attempt+1} failed: got {classification}, expected {expected_classification}"
            )

    # All attempts failed
    stats["failed"] += 1
    print(f"✗ [{index+1}/{total_entries}] All attempts failed, using REPLACE")

    result = {
        "messages": [
            {"role": "system", "content": f"Memory: {memory}"},
            {"role": "user", "content": query},
            {"role": "assistant", "content": "REPLACE"},
        ]
    }

    # Write REPLACE entry immediately
    write_entry_to_file(result, index, output_file)
    return False  # Failed


def write_entry_to_file(entry, index, output_file):
    """Write a single entry to the output file with index"""
    entry_with_index = entry.copy()
    entry_with_index["index"] = index

    with open(output_file, "a") as f:
        f.write(json.dumps(entry_with_index) + "\n")


def finalize_output_file(output_file, expected_count):
    """Sort by index, verify completeness, and remove index field"""
    print("\nFinalizing output file...")

    # Read all entries
    entries = load_jsonl(output_file)

    if not entries:
        print("Error: No entries found in output file")
        return False

    # Check for correct number of entries
    if len(entries) != expected_count:
        print(f"Error: Expected {expected_count} entries, found {len(entries)}")
        return False

    # Sort by index
    entries.sort(key=lambda x: x.get("index", float("inf")))

    # Verify all indices are present and check for REPLACE
    indices_found = set()
    has_replace = False

    for entry in entries:
        if "index" not in entry:
            print(f"Error: Entry missing index field")
            return False
        indices_found.add(entry["index"])

        # Check for REPLACE in assistant content
        if entry["messages"][2]["content"] == "REPLACE":
            has_replace = True

    # Check if all indices from 0 to expected_count-1 are present
    expected_indices = set(range(expected_count))
    if indices_found != expected_indices:
        missing = expected_indices - indices_found
        extra = indices_found - expected_indices
        if missing:
            print(f"Error: Missing indices: {sorted(missing)}")
        if extra:
            print(f"Error: Extra indices: {sorted(extra)}")
        return False

    if has_replace:
        print("Warning: Found REPLACE entries in the output")
        # Don't return False, just warn

    # Remove index field and write back in correct order
    cleaned_entries = []
    for entry in entries:
        cleaned_entry = {"messages": entry["messages"]}
        cleaned_entries.append(cleaned_entry)

    # Write back to file
    with open(output_file, "w") as f:
        for entry in cleaned_entries:
            f.write(json.dumps(entry) + "\n")

    print(
        f"✓ Successfully finalized {output_file} with {len(cleaned_entries)} entries in correct order"
    )
    return True


def main():
    input_file = "data/therapy/train.jsonl"
    output_file = "data/therapy/finetuning.jsonl"
    temp_output_file = "data/therapy/finetuning_temp.jsonl"

    print("Loading training data...")
    train_data = load_jsonl(input_file)

    if not train_data:
        print(f"Error: Could not load training data from {input_file}")
        return

    print(f"Loaded {len(train_data)} training examples")

    # Clear temp output file if it exists
    if os.path.exists(temp_output_file):
        os.remove(temp_output_file)

    # Separate gameable and non-gameable entries
    gameable_entries = [
        (i, entry) for i, entry in enumerate(train_data) if entry["is_gameable"]
    ]
    non_gameable_entries = [
        (i, entry) for i, entry in enumerate(train_data) if not entry["is_gameable"]
    ]

    print(
        f"Found {len(gameable_entries)} gameable and {len(non_gameable_entries)} non-gameable entries"
    )

    stats = {"successful": 0, "failed": 0, "total": 0}

    # Generate from scratch - write to temp file
    print("Generating finetuning data...")
    stats["total"] = len(train_data)

    # Process in pairs for efficiency
    min_pairs = min(len(gameable_entries), len(non_gameable_entries))
    print(
        f"Processing {min_pairs} pairs and {len(train_data) - 2*min_pairs} single entries..."
    )

    # Process pairs with parallel execution
    max_workers = 5  # Reduced since each call processes 2 entries

    def process_pair_wrapper(pair_idx):
        gameable_idx, gameable_entry = gameable_entries[pair_idx]
        non_gameable_idx, non_gameable_entry = non_gameable_entries[pair_idx]

        return process_entry_pair(
            gameable_entry,
            non_gameable_entry,
            gameable_idx,
            non_gameable_idx,
            len(train_data),
            stats,
            temp_output_file,  # Write to temp file
        )

    print(f"Processing {min_pairs} pairs with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit pair tasks
        future_to_pair = {
            executor.submit(process_pair_wrapper, i): i for i in range(min_pairs)
        }

        # Process pair results
        with tqdm(total=min_pairs, desc="Processing pairs") as pbar:
            for future in as_completed(future_to_pair):
                try:
                    success = future.result()
                    pbar.update(1)
                except Exception as e:
                    pair_idx = future_to_pair[future]
                    print(f"Error processing pair {pair_idx}: {e}")
                    pbar.update(1)

    # Process remaining single entries
    remaining_entries = []
    if len(gameable_entries) > min_pairs:
        remaining_entries.extend(gameable_entries[min_pairs:])
    if len(non_gameable_entries) > min_pairs:
        remaining_entries.extend(non_gameable_entries[min_pairs:])

    if remaining_entries:
        print(f"\nProcessing {len(remaining_entries)} remaining single entries...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    process_single_entry,
                    entry,
                    idx,
                    len(train_data),
                    stats,
                    temp_output_file,
                ): idx
                for idx, entry in remaining_entries
            }

            with tqdm(total=len(remaining_entries), desc="Processing singles") as pbar:
                for future in as_completed(future_to_idx):
                    try:
                        success = future.result()
                        pbar.update(1)
                    except Exception as e:
                        idx = future_to_idx[future]
                        print(f"Error processing single entry {idx}: {e}")
                        pbar.update(1)

    print(f"\nCompleted processing:")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed (marked as REPLACE): {stats['failed']}")

    # Finalize the output file - sort and verify
    if finalize_output_file(temp_output_file, len(train_data)):
        # Move temp file to final output file
        os.rename(temp_output_file, output_file)
        print(f"Successfully created {output_file}")
    else:
        print("Error: Failed to finalize output file")


if __name__ == "__main__":
    main()
