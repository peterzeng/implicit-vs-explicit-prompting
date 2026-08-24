"""
Script to extract AI-AI dialogues from experiment CSV and output to human-readable text file.
"""

import csv
import json
from pathlib import Path


def parse_messages(messages_json: str) -> list[dict]:
    """Parse the JSON messages string into a list of message dicts."""
    if not messages_json or messages_json.strip() == '[]':
        return []
    try:
        return json.loads(messages_json)
    except json.JSONDecodeError:
        return []


def format_dialogue(messages: list[dict]) -> str:
    """Format a list of messages into a readable dialogue string."""
    if not messages:
        return "  (No messages recorded)\n"
    
    lines = []
    for msg in messages:
        role = msg.get('sender_role', 'unknown').upper()
        text = msg.get('text', '')
        # Clean up the text - remove excessive markdown formatting for readability
        lines.append(f"  [{role}]:\n    {text}\n")
    
    return "\n".join(lines)


def extract_round_data(row: dict, round_num: int) -> dict:
    """Extract relevant data for a specific round."""
    prefix = f"referential_task.{round_num}"
    
    return {
        'messages': row.get(f'{prefix}.group.ai_messages', '[]'),
        'accuracy': row.get(f'{prefix}.player.sequence_accuracy', 'N/A'),
        'shared_grid': row.get(f'{prefix}.group.shared_grid', '[]'),
        'target_baskets': row.get(f'{prefix}.group.target_baskets', '[]'),
        'matcher_sequence': row.get(f'{prefix}.group.matcher_sequence', '[]'),
    }


def format_grid_summary(shared_grid_json: str) -> str:
    """Create a brief summary of the grid layout."""
    try:
        grid = json.loads(shared_grid_json) if shared_grid_json else []
        if not grid:
            return "  (No grid data)"
        
        lines = ["  Grid Layout (position -> image):"]
        for item in grid:
            pos = item.get('position', '?')
            img = item.get('image', '?').replace('images/', '')
            lines.append(f"    Position {pos}: {img}")
        return "\n".join(lines)
    except json.JSONDecodeError:
        return "  (Could not parse grid data)"


def main():
    # Paths
    input_file = Path("data/5.2-simple-prompt.csv")
    output_file = Path("data/ai_dialogues_readable.txt")
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("No data rows found in CSV.")
        return
    
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("AI-AI EXPERIMENT DIALOGUES")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # Get session info from first row
    row = rows[0]
    session_code = row.get('session.code', 'Unknown')
    director_model = row.get('session.config.ai_director_model', 'Unknown')
    matcher_model = row.get('session.config.ai_matcher_model', 'Unknown')
    prompt_strategy = row.get('session.config.prompt_strategy', 'Unknown')
    reasoning_effort = row.get('session.config.ai_reasoning_effort', 'Unknown')
    
    output_lines.append(f"Session Code: {session_code}")
    output_lines.append(f"Director Model: {director_model}")
    output_lines.append(f"Matcher Model: {matcher_model}")
    output_lines.append(f"Prompt Strategy: {prompt_strategy}")
    output_lines.append(f"Reasoning Effort: {reasoning_effort}")
    output_lines.append("")
    
    round_nums = []
    for key in row:
        prefix = "referential_task."
        if key.startswith(prefix):
            rest = key[len(prefix):]
            round_part = rest.split(".", 1)[0]
            if round_part.isdigit():
                round_nums.append(int(round_part))
    max_round = max(round_nums, default=5)

    # Process each round
    for round_num in range(1, max_round + 1):
        output_lines.append("-" * 80)
        output_lines.append(f"ROUND {round_num}")
        output_lines.append("-" * 80)
        output_lines.append("")
        
        round_data = extract_round_data(row, round_num)
        
        # Accuracy
        accuracy = round_data['accuracy']
        output_lines.append(f"Accuracy: {accuracy}%")
        output_lines.append("")
        
        # Grid summary
        output_lines.append("Grid Layout:")
        output_lines.append(format_grid_summary(round_data['shared_grid']))
        output_lines.append("")
        
        # Dialogue
        output_lines.append("Dialogue:")
        output_lines.append("-" * 40)
        messages = parse_messages(round_data['messages'])
        output_lines.append(format_dialogue(messages))
        output_lines.append("")
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))
    
    print(f"Dialogues extracted to: {output_file}")
    print(f"Processed {len(rows)} participant(s), {max_round} rounds each.")


if __name__ == "__main__":
    main()
