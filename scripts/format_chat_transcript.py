"""
Format chat transcript from single-line to multi-line format.

Usage:
    python scripts/format_chat_transcript.py input.txt output.txt
    
Or process all chat columns from a CSV:
    python scripts/format_chat_transcript.py data.csv output_folder/
"""

import re
import sys
import os
import csv
from pathlib import Path


def format_chat_transcript(raw_text):
    """
    Convert single-line chat transcript to multi-line format.
    
    Input:  "[15:59:54] matcher: I am peter  [15:59:57] director: i am groot"
    Output: "[15:59:54] matcher: I am peter\n[15:59:57] director: i am groot"
    """
    if not raw_text or not raw_text.strip():
        return ""
    
    # Pattern to match timestamps like [HH:MM:SS] or [HH:MM:SS.mmm]
    # This splits before each timestamp
    pattern = r'(\[\d{2}:\d{2}:\d{2}(?:\.\d+)?\])'
    
    # Split by the pattern but keep the delimiters (timestamps)
    parts = re.split(pattern, raw_text)
    
    # Reconstruct with newlines
    lines = []
    i = 1  # Start at 1 to skip the first empty string
    while i < len(parts):
        if i + 1 < len(parts):
            # Combine timestamp with its message
            timestamp = parts[i]
            message = parts[i + 1].strip()
            if message:
                lines.append(f"{timestamp} {message}")
            i += 2
        else:
            i += 1
    
    return '\n'.join(lines)


def process_single_file(input_path, output_path):
    """Process a single text file with chat transcript."""
    print(f"Reading from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    formatted = format_chat_transcript(raw_text)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(formatted)
    
    print(f"Formatted transcript saved to: {output_path}")
    print(f"Lines: {len(formatted.splitlines())}")


def process_csv_file(input_csv, output_folder):
    """Process all chat transcript columns from a CSV file."""
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    print(f"Reading CSV: {input_csv}")
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Find all chat_transcript columns
    if not rows:
        print("No data found in CSV")
        return
    
    chat_columns = [col for col in rows[0].keys() if 'chat_transcript' in col.lower() or 'grid_messages' in col.lower()]
    
    if not chat_columns:
        print("No chat transcript columns found in CSV")
        print(f"Available columns: {', '.join(rows[0].keys()[:10])}...")
        return
    
    print(f"Found chat columns: {', '.join(chat_columns)}")
    
    # Process each row
    for idx, row in enumerate(rows, 1):
        participant_code = row.get('participant.code', f'participant_{idx}')
        round_num = row.get('subsession.round_number', idx)
        
        for col in chat_columns:
            raw_text = row.get(col, '')
            if raw_text and raw_text.strip():
                formatted = format_chat_transcript(raw_text)
                
                # Create filename
                col_short = col.split('.')[-1] if '.' in col else col
                filename = f"{participant_code}_round{round_num}_{col_short}.txt"
                output_path = output_folder / filename
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(formatted)
                
                print(f"  Saved: {filename} ({len(formatted.splitlines())} lines)")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nExample:")
        print("  python scripts/format_chat_transcript.py chat.json formatted_chat.txt")
        print("  python scripts/format_chat_transcript.py data.csv output_chats/")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)
    
    # Check if input is CSV
    if input_path.endswith('.csv'):
        process_csv_file(input_path, output_path)
    else:
        process_single_file(input_path, output_path)


if __name__ == '__main__':
    main()

