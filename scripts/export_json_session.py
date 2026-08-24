"""
Export JSON session data to visual comparison grids and transcript text file.

Usage:
    python scripts/export_json_session.py data/Cameron-Test-Serious_data.json
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import sys
# Ensure we can import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.visualize_matcher_sequence import create_combined_visualization, compare_sequences

def format_timestamp(ts_str):
    if not ts_str:
        return "00:00:00"
    # Try different ts formats
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.strftime('%H:%M:%S')
    except ValueError:
        return ts_str

def calculate_accuracy(shared_grid, matcher_sequence):
    if not shared_grid:
        return 0.0
    errors = compare_sequences(shared_grid, matcher_sequence)
    total = len(shared_grid)
    if total == 0:
        return 0.0
    return round(((total - len(errors)) / total) * 100, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Path to JSON file")
    parser.add_argument(
        "--output_dir",
        "-o",
        default=None,
        help="Directory for output artifacts. Defaults to the input JSON's directory.",
    )
    args = parser.parse_args()

    # Load JSON
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        print("Empty JSON data in file.")
        return

    input_path = Path(args.input_json)

    # Use session_id from the first round, or fallback
    session_id = data[0].get('session_id', 'unknown_session')
    base_name = input_path.stem.replace('_data', '')
    
    # By default, save transcript/images next to the source JSON so all artifacts
    # for a run can be moved together. Prefix filenames to avoid overwriting when
    # multiple session JSONs share one experiment folder.
    out_folder = Path(args.output_dir) if args.output_dir else input_path.parent
    out_folder.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to directory: {out_folder}")

    # Prepare transcript report headers
    transcript_lines = []
    transcript_lines.append("=" * 80)
    transcript_lines.append("SESSION TRANSCRIPT REPORT")
    transcript_lines.append("=" * 80)
    transcript_lines.append("")
    transcript_lines.append(f"Pair ID:        {session_id}")
    transcript_lines.append(f"Session Code:   {session_id}")
    transcript_lines.append(f"Source File:    {args.input_json}")
    
    # Optional conf details if config object exists
    config = data[0].get('config', {})
    if config:
        transcript_lines.append("")
        transcript_lines.append("SESSION CONFIGURATION:")
        for k, v in config.items():
            transcript_lines.append(f"  {k}: {v}")
    
    # Calculate overall accuracy
    accuracies = []
    for round_obj in data:
        shared_grid = round_obj.get('shared_grid', [])
        matcher_seq = round_obj.get('matcher_sequence', [])
        acc = calculate_accuracy(shared_grid, matcher_seq)
        if len(shared_grid) > 0 and len(matcher_seq) > 0:
            accuracies.append(acc)
    
    overall_acc = round(sum(accuracies)/len(accuracies), 1) if accuracies else 0.0
    transcript_lines.append("")
    transcript_lines.append(f"OVERALL ACCURACY: {overall_acc}% (avg across {len(data)} rounds)")
    transcript_lines.append("")
    transcript_lines.append("=" * 80)
    
    # Use standard static images directory for visualizations
    images_dir = "_static/images"

    for idx, round_obj in enumerate(data):
        round_num = round_obj.get('round_number', idx + 1)
        shared_grid = round_obj.get('shared_grid', [])
        matcher_sequence = round_obj.get('matcher_sequence', [])
        ai_messages = round_obj.get('ai_messages', [])
        
        acc = calculate_accuracy(shared_grid, matcher_sequence)
        
        # Add to Transcript report
        transcript_lines.append("")
        transcript_lines.append(f"ROUND {round_num}")
        transcript_lines.append("-" * 40)
        transcript_lines.append(f"Accuracy: {acc}%")
        transcript_lines.append("")
        transcript_lines.append("DIALOGUE:")
        transcript_lines.append("-" * 20)
        
        for msg in ai_messages:
            role = msg.get('sender_role', 'unknown').upper()
            ts = format_timestamp(msg.get('timestamp', ''))
            text = msg.get('text', '')
            transcript_lines.append(f"[{ts}] {role}: {text}")
            transcript_lines.append("")
        
        transcript_lines.append("=" * 80)
        
        # Generate Visualization image
        round_data_for_viz = {
            'shared_grid': shared_grid,
            'matcher_sequence': matcher_sequence,
            'accuracy': acc
        }
        
        # Default empty output if no grid or seq exists
        if shared_grid and matcher_sequence:
            create_combined_visualization(
                round_num,
                round_data_for_viz,
                images_dir,
                str(out_folder),
                output_filename=f"{base_name}_round_{round_num}_comparison.png",
            )
        else:
            print(f"Skipping visualization for Round {round_num}: missing sequence or grid data.")

    # Save the transcript
    transcript_path = out_folder / f"{base_name}_transcript.txt"
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(transcript_lines))

    print(f"Export complete. Check {out_folder}")

if __name__ == "__main__":
    main()
