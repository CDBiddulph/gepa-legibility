#!/usr/bin/env python3
"""
Process JSONL files to rename fields for RunRL compatibility.
Outputs to a new file with 'runrl-' prefix in the same directory.
"""

import argparse
import json
import os
from pathlib import Path


def process_jsonl(input_file: str, old_field: str, new_field: str, output_file: str):
    """
    Process a JSONL file, renaming a specified field and converting to message format.

    Args:
        input_file: Path to input JSONL file
        old_field: Name of field to rename
        new_field: New name for the field
        output_file: Path to output JSONL file
    """
    processed_count = 0
    renamed_count = 0

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Rename the field if it exists
                if old_field in data:
                    old_value = data.pop(old_field)
                    # Convert to message format with role "user"
                    data[new_field] = [{"role": "user", "content": old_value}]
                    renamed_count += 1

                # Write to output file
                outfile.write(json.dumps(data) + '\n')
                processed_count += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue

    print(f"Processed {processed_count} lines")
    print(f"Renamed field '{old_field}' -> '{new_field}' in {renamed_count} lines")
    print(f"Output written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Rename fields in JSONL files for RunRL compatibility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data/wordchain/train.jsonl --old-field query --new-field prompt
  %(prog)s -i data/wordchain/train.jsonl -o query -n prompt
        """
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input JSONL file path'
    )

    parser.add_argument(
        '--old-field', '-o',
        required=True,
        help='Name of the field to rename (e.g., "query")'
    )

    parser.add_argument(
        '--new-field', '-n',
        required=True,
        help='New name for the field (e.g., "prompt")'
    )

    parser.add_argument(
        '--output',
        help='Output file path (default: same directory as input with "runrl-" prefix)'
    )

    args = parser.parse_args()

    # Determine output file path
    if args.output:
        output_file = args.output
    else:
        input_path = Path(args.input)
        output_filename = f"runrl-{input_path.name}"
        output_file = str(input_path.parent / output_filename)

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return 1

    # Process the file
    process_jsonl(args.input, args.old_field, args.new_field, output_file)

    return 0


if __name__ == '__main__':
    exit(main())
