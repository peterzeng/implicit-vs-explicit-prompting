import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict

def flatten_log_to_csv(input_json_path: str, output_csv_path: str):
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Fix any trailing garbage by safely truncating at the last bracket
            if ']' in content:
                content = content[:content.rindex(']') + 1]
            data = json.loads(content)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    # Group rounds by session_id
    sessions = defaultdict(list)
    for row in data:
        sid = row.get("session_id", "unknown")
        sessions[sid].append(row)

    def safe_round_number(row):
        try:
            return int(row.get("round_number", 0) or 0)
        except (TypeError, ValueError):
            return 0

    max_rounds = max((safe_round_number(row) for row in data), default=0)

    headers = ["session_id"]
    for round_num in range(1, max_rounds + 1):
        headers.extend([
            f"round_{round_num}_conversation",
            f"round_{round_num}_ai_reasoning",
            f"round_{round_num}_accuracy",
        ])
    headers.extend([
        "director_perception_capable",
        "director_perception_helpful",
        "director_perception_understood",
        "director_perception_adapted",
        "director_perception_improved",
        "director_perception_comment",
        "matcher_perception_capable",
        "matcher_perception_helpful",
        "matcher_perception_understood",
        "matcher_perception_adapted",
        "matcher_perception_improved",
        "matcher_perception_comment"
    ])

    rows = []

    for sid, rounds in sessions.items():
        # Sort rounds safely
        rounds.sort(key=lambda r: int(r.get("round_number", 0)))
        
        row_data = {
            "session_id": sid,
            "director_perception_capable": "",
            "director_perception_helpful": "",
            "director_perception_understood": "",
            "director_perception_adapted": "",
            "director_perception_improved": "",
            "director_perception_comment": "",
            "matcher_perception_capable": "",
            "matcher_perception_helpful": "",
            "matcher_perception_understood": "",
            "matcher_perception_adapted": "",
            "matcher_perception_improved": "",
            "matcher_perception_comment": ""
        }
        
        for rnd in rounds:
            r_num = str(rnd.get("round_number", ""))
            status = rnd.get("status", {})
            accuracy = status.get("accuracy", "")
            
            messages = status.get("messages", [])
            conv_str = ""
            for msg in messages:
                tstamp = msg.get("timestamp", "")
                role = msg.get("sender_role", "").upper()
                text = msg.get("text", "")
                conv_str += f"[{tstamp}] {role}:\n{text}\n\n"
                
            row_data[f"round_{r_num}_conversation"] = conv_str.strip()
            row_data[f"round_{r_num}_accuracy"] = accuracy
            
            reasoning_log = rnd.get("ai_reasoning_log", [])
            reasoning_str = ""
            for r in reasoning_log:
                role_str = r.get("ai_role", "unknown").upper()
                ts = r.get("timestamp", "")
                reasoning_obj = r.get("reasoning")
                
                if reasoning_obj:
                    # Output the nicely formatted reasoning dictionary if available
                    body = json.dumps(reasoning_obj, indent=2)
                else:
                    # Fallback to the raw text
                    body = r.get("raw_text", "")
                    
                reasoning_str += f"[{ts}] {role_str} REASONING:\n{body}\n\n"
                
            row_data[f"round_{r_num}_ai_reasoning"] = reasoning_str.strip()
            
            # Perceptions exist primarily at the end of the final round
            d_reasoning = rnd.get("ai_director_reasoning")
            if d_reasoning:
                row_data["director_perception_capable"] = d_reasoning.get("partner_capable", "")
                row_data["director_perception_helpful"] = d_reasoning.get("partner_helpful", "")
                row_data["director_perception_understood"] = d_reasoning.get("partner_understood", "")
                row_data["director_perception_adapted"] = d_reasoning.get("partner_adapted", "")
                row_data["director_perception_improved"] = d_reasoning.get("collaboration_improved", "")
                row_data["director_perception_comment"] = d_reasoning.get("partner_comment", "")
                
            m_reasoning = rnd.get("ai_matcher_reasoning")
            if m_reasoning:
                row_data["matcher_perception_capable"] = m_reasoning.get("partner_capable", "")
                row_data["matcher_perception_helpful"] = m_reasoning.get("partner_helpful", "")
                row_data["matcher_perception_understood"] = m_reasoning.get("partner_understood", "")
                row_data["matcher_perception_adapted"] = m_reasoning.get("partner_adapted", "")
                row_data["matcher_perception_improved"] = m_reasoning.get("collaboration_improved", "")
                row_data["matcher_perception_comment"] = m_reasoning.get("partner_comment", "")

        out_row = []
        for h in headers:
            out_row.append(row_data.get(h, ""))
            
        rows.append(out_row)

    try:
        with open(output_csv_path, 'w', newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"Successfully wrote flattened session CSV to: {output_csv_path}")
    except Exception as e:
        print(f"Error writing CSV: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format referential game JSON logs into 1-row-per-session CSV.")
    parser.add_argument("input_file", help="Path to the source _data.json file")
    parser.add_argument("--output", "-o", help="Path to exactly save the output csv. If omitted, derives from input file name.")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"File not found: {input_path}")
        exit(1)
        
    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.with_name(f"{input_path.stem}_session_formatted.csv"))
        
    flatten_log_to_csv(str(input_path), output_path)
