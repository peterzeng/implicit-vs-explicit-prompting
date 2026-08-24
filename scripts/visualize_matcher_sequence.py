"""
Script to visualize the AI matcher's submitted sequences for each round.
Recreates the grid images showing Director's target vs Matcher's sequence.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Default Configuration
DEFAULT_CSV_FILE = "data/5.2-simple-prompt.csv"
IMAGES_DIR = "_static/images"
OUTPUT_DIR = "data/sequence_visualizations"

# Grid layout
GRID_COLS = 4
GRID_ROWS = 3
CELL_SIZE = 150
PADDING = 10
LABEL_HEIGHT = 40


def load_csv_data(csv_path):
    """Load and parse the CSV file."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows[0] if rows else None  # Single participant row


def parse_json_field(field_value):
    """Parse a JSON field from the CSV."""
    if not field_value or field_value.strip() == '':
        return []
    try:
        return json.loads(field_value)
    except json.JSONDecodeError:
        return []


def get_round_data(row, round_num):
    """Extract grid data for a specific round."""
    prefix = f"referential_task.{round_num}.group."
    
    shared_grid = parse_json_field(row.get(f"{prefix}shared_grid", ""))
    matcher_sequence = parse_json_field(row.get(f"{prefix}matcher_sequence", ""))
    target_baskets = parse_json_field(row.get(f"{prefix}target_baskets", ""))
    accuracy = row.get(f"referential_task.{round_num}.player.sequence_accuracy", "")
    
    return {
        'shared_grid': shared_grid,
        'matcher_sequence': matcher_sequence,
        'target_baskets': target_baskets,
        'accuracy': accuracy
    }


def normalize_target_sequence(target_seq):
    """Return target baskets in reading order with explicit positions 1..12."""
    normalized = []
    for idx, item in enumerate(sorted(target_seq, key=lambda x: (x.get('row', 0), x.get('col', 0))), start=1):
        slot = dict(item)
        slot['position'] = idx
        normalized.append(slot)
    return normalized


def normalize_matcher_sequence(matcher_seq, total_positions=12):
    """Return matcher baskets indexed by sequence position, preserving blanks."""
    by_pos = {}
    for item in matcher_seq or []:
        try:
            pos = int(item.get('position'))
        except (TypeError, ValueError, AttributeError):
            continue
        if 1 <= pos <= total_positions:
            by_pos[pos] = item

    normalized = []
    for pos in range(1, total_positions + 1):
        if pos in by_pos:
            slot = dict(by_pos[pos])
            slot['position'] = pos
        else:
            slot = {'position': pos, 'image': None, 'originalPosition': None}
        normalized.append(slot)
    return normalized


def create_grid_image(sequence, images_dir, title, highlight_errors=None):
    """Create a grid image from a sequence of basket data."""
    width = GRID_COLS * (CELL_SIZE + PADDING) + PADDING
    height = GRID_ROWS * (CELL_SIZE + PADDING) + PADDING + LABEL_HEIGHT
    
    # Create image with dark background
    img = Image.new('RGB', (width, height), color=(40, 44, 52))
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        small_font = font
    
    # Draw title
    draw.text((PADDING, 5), title, fill=(255, 255, 255), font=font)
    
    # Sort sequence by explicit sequence position. Matcher sequences may contain
    # blanks, so callers should preserve those as placeholder items.
    sorted_seq = sorted(sequence, key=lambda x: x.get('position', 0))
    
    for idx, item in enumerate(sorted_seq):
        row = idx // GRID_COLS
        col = idx % GRID_COLS
        
        x = col * (CELL_SIZE + PADDING) + PADDING
        y = row * (CELL_SIZE + PADDING) + PADDING + LABEL_HEIGHT
        
        # Determine border color
        border_color = (100, 100, 100)  # Default gray
        if highlight_errors is not None:
            if idx in highlight_errors:
                border_color = (220, 50, 50)  # Red for errors
            else:
                border_color = (50, 180, 50)  # Green for correct
        
        # Draw cell border
        draw.rectangle([x-2, y-2, x+CELL_SIZE+2, y+CELL_SIZE+2], outline=border_color, width=3)
        
        # Load and paste basket image
        image_path = item.get('image', '')
        if image_path:
            # Handle both "images/XXX.png" and just "XXX.png" formats
            if image_path.startswith('images/'):
                image_path = image_path.replace('images/', '')
            
            full_path = os.path.join(images_dir, image_path)
            if os.path.exists(full_path):
                try:
                    basket_img = Image.open(full_path)
                    basket_img = basket_img.resize((CELL_SIZE, CELL_SIZE), Image.Resampling.LANCZOS)
                    img.paste(basket_img, (x, y))
                except Exception as e:
                    print(f"Error loading image {full_path}: {e}")
                    draw.rectangle([x, y, x+CELL_SIZE, y+CELL_SIZE], fill=(80, 80, 80))
            else:
                # Draw placeholder
                draw.rectangle([x, y, x+CELL_SIZE, y+CELL_SIZE], fill=(80, 80, 80))
                draw.text((x+10, y+60), "Missing", fill=(200, 200, 200), font=small_font)
        
        # Draw position number
        pos_label = str(idx + 1)
        draw.rectangle([x, y, x+25, y+20], fill=(0, 0, 0, 180))
        draw.text((x+5, y+2), pos_label, fill=(255, 255, 0), font=small_font)
    
    return img


def compare_sequences(target_seq, matcher_seq):
    """Compare sequences and return indices of errors."""
    errors = set()

    target_items = normalize_target_sequence(target_seq)
    matcher_items = normalize_matcher_sequence(matcher_seq, len(target_items))
    target_images = [item.get('image', '') for item in target_items]
    matcher_images = [item.get('image', '') for item in matcher_items]
    
    # Compare
    for i in range(len(target_images)):
        if target_images[i] != matcher_images[i]:
            errors.add(i)
    
    return errors


def create_combined_visualization(round_num, round_data, images_dir, output_dir, output_filename=None):
    """Create a combined visualization for a round."""
    shared_grid = round_data['shared_grid']
    matcher_sequence = round_data['matcher_sequence']
    accuracy = round_data['accuracy']
    
    if not shared_grid or not matcher_sequence:
        print(f"  Round {round_num}: No data available")
        return None
    
    # Find errors
    errors = compare_sequences(shared_grid, matcher_sequence)
    
    target_sequence = normalize_target_sequence(shared_grid)
    matcher_sequence = normalize_matcher_sequence(matcher_sequence, len(target_sequence))

    # Create director's grid (target)
    director_title = "DIRECTOR'S TARGET SEQUENCE"
    director_img = create_grid_image(target_sequence, images_dir, director_title)
    
    # Create matcher's grid (with error highlighting)
    matcher_title = f"MATCHER'S SEQUENCE ({accuracy}% accuracy)"
    matcher_img = create_grid_image(matcher_sequence, images_dir, matcher_title, highlight_errors=errors)
    
    # Combine side by side
    gap = 40
    combined_width = director_img.width + matcher_img.width + gap
    combined_height = max(director_img.height, matcher_img.height) + 60
    
    combined = Image.new('RGB', (combined_width, combined_height), color=(30, 30, 35))
    draw = ImageDraw.Draw(combined)
    
    # Try to load a font
    try:
        title_font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        title_font = ImageFont.load_default()
    
    # Draw round title
    round_title = f"Round {round_num}"
    draw.text((combined_width // 2 - 50, 10), round_title, fill=(255, 255, 255), font=title_font)
    
    # Paste grids
    combined.paste(director_img, (0, 50))
    combined.paste(matcher_img, (director_img.width + gap, 50))
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        output_filename or f"round_{round_num}_comparison.png",
    )
    combined.save(output_path)
    print(f"  Saved: {output_path}")
    
    return output_path


def get_output_subfolder(csv_file_path, subfolder_name=None):
    """Generate output subfolder path based on CSV file name or custom name."""
    if subfolder_name:
        return os.path.join(OUTPUT_DIR, subfolder_name)
    
    # Extract base name from CSV file (without extension)
    csv_basename = Path(csv_file_path).stem
    return os.path.join(OUTPUT_DIR, csv_basename)


def main():
    """Main function to generate all visualizations."""
    parser = argparse.ArgumentParser(
        description="Visualize AI matcher's submitted sequences for each round"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=DEFAULT_CSV_FILE,
        help=f"Path to input CSV file (default: {DEFAULT_CSV_FILE})"
    )
    parser.add_argument(
        "--subfolder",
        "-s",
        type=str,
        default=None,
        help="Name of subfolder in output directory (default: uses CSV filename)"
    )
    parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=4,
        help="Number of rounds to process (default: 4)"
    )
    
    args = parser.parse_args()
    
    csv_file = args.input
    subfolder_name = args.subfolder
    num_rounds = args.rounds
    
    # Create output subfolder path
    output_subfolder = get_output_subfolder(csv_file, subfolder_name)
    
    print(f"Input CSV: {csv_file}")
    print(f"Output directory: {output_subfolder}")
    print("Loading CSV data...")
    
    row = load_csv_data(csv_file)
    
    if not row:
        print("Error: No data found in CSV")
        return
    
    print(f"Processing {num_rounds} rounds...\n")
    
    for round_num in range(1, num_rounds + 1):
        print(f"Round {round_num}:")
        round_data = get_round_data(row, round_num)
        
        if round_data['shared_grid']:
            create_combined_visualization(round_num, round_data, IMAGES_DIR, output_subfolder)
        else:
            print(f"  No data for round {round_num}")
    
    print(f"\nDone! Visualizations saved to: {output_subfolder}/")


if __name__ == "__main__":
    main()
