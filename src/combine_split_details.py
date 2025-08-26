#!/usr/bin/env python3
import os
import argparse

# directories that represent sampling methods
SAMPLING_DIRS = {'original', 'oversample', 'undersample'}

def combine_split_details(base_dir: str, output_path: str):
    """
    Walk base_dir, find every split_details.txt under
    original/, oversample/, undersample/,
    then write into output_path with headers:
      Sampling method, User, Dataset, Pipeline,
    and include only the summary section before any day breakdowns.
    """
    files_found = 0
    with open(output_path, 'w') as outfile:
        for root, dirs, files in os.walk(base_dir):
            if 'split_details.txt' not in files:
                continue

            parts = root.split(os.sep)
            # locate which sampling method folder we're under
            samp_idx = next((i for i, p in enumerate(parts) if p in SAMPLING_DIRS), None)
            if samp_idx is None or len(parts) < samp_idx + 4:
                continue

            sampling_method = parts[samp_idx]
            user            = parts[samp_idx + 1]
            dataset         = parts[samp_idx + 2]
            pipeline        = parts[samp_idx + 3]

            file_path = os.path.join(root, 'split_details.txt')
            with open(file_path, 'r') as infile:
                lines = infile.readlines()

            # keep only lines before any '=== TRAINING/VALIDATION/TEST' sections
            summary_lines = []
            for line in lines:
                if line.strip().startswith('==='):
                    break
                summary_lines.append(line)
            content = ''.join(summary_lines).rstrip()

            header = (
                f"\n{'='*30}\n"
                f"Sampling method: {sampling_method}\n"
                f"User:            {user}\n"
                f"Dataset:         {dataset}\n"
                f"Pipeline:        {pipeline}\n"
                f"{'='*30}\n\n"
            )

            outfile.write(header)
            outfile.write(content + "\n")
            files_found += 1

    print(f"✔ Combined {files_found} split_details.txt files into {output_path!r}")


if __name__ == '__main__':
    # assume script lives in src/, project root one level up
    PROJECT_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DEFAULT_BASE    = PROJECT_ROOT
    DEFAULT_OUTPUT  = os.path.join(PROJECT_ROOT, 'all_split_details.txt')

    parser = argparse.ArgumentParser(
        description="Combine every split_details.txt from original/oversample/undersample into one file without detailed day breakdowns."
    )
    parser.add_argument('-b', '--base-dir',
        default=DEFAULT_BASE,
        help=f"root folder to search (default: {DEFAULT_BASE})"
    )
    parser.add_argument('-o', '--output',
        default=DEFAULT_OUTPUT,
        help=f"path for combined output (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()
    combine_split_details(args.base_dir, args.output)
