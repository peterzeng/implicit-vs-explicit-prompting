#!/usr/bin/env python3
from __future__ import annotations

"""
Knowledge Base Generator for Basket Referential Game (oTree grid-ai branch)

This is adapted from the main Human-VLM-Game repo (Weiling's GPTMatchingBot).
It analyzes individual basket images and generates a JSON knowledge base with
detailed descriptions of each basket's features for use in prompting.

Usage (from project root):

    conda activate langviscog  # or your env
    pip install -r requirements.txt

    python scripts/generate_knowledge_base.py \
        --image-dir _static/images \
        --output referential_task/KnowledgeBase.json \
        --model gpt-4o

The generated JSON file can then be loaded at runtime by the oTree app.
"""

import json
import os
import sys
from typing import Dict, Any

from litellm import completion  # type: ignore
import argparse
from pathlib import Path
try:
    from dotenv import load_dotenv  # type: ignore
    # project root is the parent of this scripts/ directory
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
    load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))
except Exception:
    pass

class BasketAnalyzer:
    def __init__(self, model_name: str = "gpt-4o", api_key: str | None = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass it directly."
            )

        # Set the API key for litellm
        os.environ["OPENAI_API_KEY"] = self.api_key

    def analyze_basket_image(self, image_path: str, basket_id: int) -> Dict[str, Any]:
        """Analyze a single basket image and extract comprehensive features."""

        # Convert to absolute path and encode as base64 for local images
        import base64

        abs_path = os.path.abspath(image_path)

        # Read and encode the image as base64
        try:
            with open(abs_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")

            # Determine image type from file extension
            file_extension = Path(abs_path).suffix.lower()
            if file_extension == ".png":
                image_type = "image/png"
            elif file_extension in [".jpg", ".jpeg"]:
                image_type = "image/jpeg"
            else:
                image_type = "image/png"  # Default to PNG

            image_url = f"data:{image_type};base64,{image_data}"

        except Exception as e:
            print(f"Error reading image file {abs_path}: {e}")
            return {
                "basket_id": basket_id,
                "error": f"Failed to read image file: {str(e)}",
                "analysis_failed": True,
            }

        analysis_prompt = f"""Analyze this basket image (ID: {basket_id}) and provide a comprehensive description of ALL observable features. Be extremely detailed and consistent in your analysis.

Please analyze and describe the following aspects in detail:

1. **Shape**: Exact shape description (round, oval, rectangular, square, conical, etc.)
2. **Color**: Primary and secondary colors, color patterns, gradients
3. **Material**: Type of material (wicker, plastic, metal, fabric, etc.)
4. **Texture**: Surface texture (smooth, rough, woven, braided, etc.)
5. **Pattern**: Any visible patterns (solid, striped, checkered, geometric, etc.)
6. **Size**: Relative size compared to typical baskets (small, medium, large)
7. **Handles**: Handle type, number, position, material, shape
8. **Lid**: Presence of lid, lid type, how it attaches
9. **Decorations**: Any decorative elements, embellishments, ornaments
10. **Structure**: Overall construction details, rim style, base type

Provide your response in JSON format with the following structure:
{{
    "basket_id": {basket_id},
    "shape": "detailed shape description",
    "color": "detailed color description",
    "material": "material description",
    "texture": "texture description",
    "pattern": "pattern description",
    "size": "size description",
    "handles": "handle description",
    "lid": "lid description",
    "decorations": "decoration description",
    "structure": "structural details",
    "distinctive_features": ["list", "of", "most", "unique", "identifying", "features"],
    "comprehensive_description": "A complete one-sentence description combining all features"
}}

Be consistent in terminology - use the same words for the same features across different baskets."""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": analysis_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            }
        ]

        try:
            response = completion(
                model=self.model_name,
                messages=messages,
                max_tokens=800,
                temperature=0.1,  # Low temperature for consistency
            )

            response_text = response.choices[0].message.content.strip()

            # Extract JSON from response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")

        except Exception as e:
            print(f"Error analyzing basket {basket_id}: {str(e)}")
            return {
                "basket_id": basket_id,
                "error": str(e),
                "analysis_failed": True,
            }

    def generate_knowledge_base(
        self, image_directory: str, output_file: str = "KnowledgeBase.json"
    ) -> Dict[str, Any]:
        """Generate comprehensive knowledge base from all basket images in directory."""

        image_dir = Path(image_directory)
        if not image_dir.exists():
            raise ValueError(f"Directory {image_directory} does not exist")

        # Find all image files (assuming they follow the pattern 001.png, 002.png, etc.)
        image_files = sorted([f for f in image_dir.glob("*.png") if f.stem.isdigit()])

        if not image_files:
            raise ValueError(f"No numbered image files found in {image_directory}")

        print(f"Found {len(image_files)} basket images to analyze...")

        knowledge_base: Dict[str, Any] = {
            "metadata": {
                "total_baskets": len(image_files),
                "model_used": self.model_name,
                "generation_info": (
                    "Generated by BasketAnalyzer for consistent basket descriptions"
                ),
            },
            "baskets": {},
        }

        for i, image_file in enumerate(image_files, 1):
            basket_id = int(image_file.stem)
            print(f"Analyzing basket {basket_id} ({i}/{len(image_files)})...")

            analysis = self.analyze_basket_image(str(image_file), basket_id)
            knowledge_base["baskets"][str(basket_id)] = analysis

            # Add a small delay to avoid rate limiting
            import time

            time.sleep(0.5)

        # Save knowledge base to file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=False)

        print("\nKnowledge base generated successfully!")
        print(f"Analyzed {len(image_files)} baskets")
        print(f"Output saved to: {output_file}")

        return knowledge_base


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Knowledge Base for Basket Referential Game (oTree grid-ai)"
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing numbered basket images (e.g., _static/images)",
    )
    parser.add_argument(
        "--output",
        default="KnowledgeBase.json",
        help="Output JSON file (default: KnowledgeBase.json)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model to use for analysis (default: gpt-4o)",
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )

    args = parser.parse_args()

    try:
        analyzer = BasketAnalyzer(model_name=args.model, api_key=args.api_key)
        knowledge_base = analyzer.generate_knowledge_base(args.image_dir, args.output)

        # Print summary
        successful_analyses = sum(
            1
            for basket in knowledge_base["baskets"].values()
            if not basket.get("analysis_failed", False)
        )
        print("\nSummary:")
        print(f"- Successfully analyzed: {successful_analyses} baskets")
        print(
            f"- Failed analyses: {len(knowledge_base['baskets']) - successful_analyses}"
        )
        print(f"- Knowledge base saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


