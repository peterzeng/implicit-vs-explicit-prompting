#!/usr/bin/env python3
"""
Test script to generate a preview of the AI Director's grid composite image.

This script creates a standalone version of the director grid generator
to preview what GPT-4o would see without requiring the full oTree stack.

Usage:
    python scripts/test_director_grid.py

Output:
    - Saves to _static/ai_debug/ai_director_grid_TEST.png
    - Opens the image automatically (on macOS)
"""
from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFont

# Project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_font(size: int = 16):
    """Load a TrueType font with fallback to default."""
    font_candidates = [
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/Library/Fonts/Arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _get_text_dimensions(draw, text, font):
    """Get text dimensions using modern Pillow API with fallback."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        try:
            return draw.textsize(text, font=font)
        except Exception:
            return (len(text) * 8, 14)


def _draw_label_badge(draw, img_canvas, text, center_x, center_y, font, bg_color, text_color, padding=6, min_width=28):
    """Draw a text label with a rounded rectangle background badge."""
    tw, th = _get_text_dimensions(draw, text, font)
    badge_w = max(tw + padding * 2, min_width)
    badge_h = th + padding * 2

    x0 = center_x - badge_w // 2
    y0 = center_y - badge_h // 2
    x1 = x0 + badge_w
    y1 = y0 + badge_h

    radius = min(8, badge_h // 2)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=bg_color)

    tx = x0 + (badge_w - tw) // 2
    ty = y0 + (badge_h - th) // 2
    draw.text((tx, ty), text, font=font, fill=text_color)


def _resolve_static_image_path(rel_path: str) -> str | None:
    """Resolve a static image path to filesystem path."""
    if not rel_path:
        return None
    rel_path = rel_path.lstrip("/ ")
    path = os.path.join(project_root, "_static", rel_path)
    if os.path.exists(path):
        return path
    return None


def build_ai_director_grid_composite(shared_grid: list[dict]) -> Image.Image:
    """
    Render a 2×6 grid image showing the 12 target baskets the AI director must describe.
    
    This is a standalone version of _build_ai_director_grid_composite for testing.
    """
    # Grid geometry
    COLS = 6
    ROWS = 2
    TILE_W = 220
    TILE_H = 220
    PADDING = 12
    HEADER_H = 50
    INSTRUCTION_H = 36
    canvas_w = COLS * TILE_W + (COLS + 1) * PADDING
    grid_height = ROWS * TILE_H + (ROWS + 1) * PADDING
    canvas_h = PADDING + HEADER_H + INSTRUCTION_H + grid_height + PADDING

    # Color scheme
    bg_color = (240, 242, 245)
    slot_bg = (255, 255, 255)
    border_color = (70, 130, 180)
    header_color = (30, 40, 50)
    badge_bg = (70, 130, 180)
    badge_text = (255, 255, 255)
    instruction_color = (80, 90, 100)

    img_canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(img_canvas)

    # Load fonts
    font_header = _load_font(24)
    font_label = _load_font(18)
    font_instruction = _load_font(14)

    # Header
    heading = "TARGET SEQUENCE TO DESCRIBE (Slots 1–12)"
    if font_header:
        draw.text((PADDING + 4, PADDING + 10), heading, font=font_header, fill=header_color)

    # Instruction
    instruction = "Describe in order: top row (1->6) then bottom row (7->12). Focus on one basket at a time."
    if font_instruction:
        draw.text((PADDING + 4, PADDING + HEADER_H + 4), instruction, font=font_instruction, fill=instruction_color)

    # Grid
    grid_origin_y = PADDING + HEADER_H + INSTRUCTION_H
    for logical_pos in range(1, 13):
        row = (logical_pos - 1) // COLS
        col = (logical_pos - 1) % COLS

        x0 = PADDING + col * (TILE_W + PADDING)
        y0 = grid_origin_y + row * (TILE_H + PADDING)
        x1 = x0 + TILE_W
        y1 = y0 + TILE_H

        draw.rectangle([x0, y0, x1, y1], fill=slot_bg, outline=border_color, width=3)

        slot_idx = logical_pos - 1
        if slot_idx < len(shared_grid):
            slot = shared_grid[slot_idx]
            rel_img_path = (slot.get("image") or "").lstrip("/ ")
            if rel_img_path:
                fs_path = _resolve_static_image_path(rel_img_path)
                if fs_path and os.path.exists(fs_path):
                    try:
                        with Image.open(fs_path) as basket_img:
                            basket_img = basket_img.convert("RGB")
                            margin = 14
                            target_w = max(1, TILE_W - 2 * margin)
                            target_h = max(1, TILE_H - 2 * margin - 30)
                            basket_img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
                            bw, bh = basket_img.size
                            bx = x0 + (TILE_W - bw) // 2
                            by = y0 + (TILE_H - bh) // 2 - 12
                            img_canvas.paste(basket_img, (bx, by))
                    except Exception as e:
                        print(f"  Warning: Could not load {fs_path}: {e}")

        # Slot label
        label = f"Slot {logical_pos}"
        badge_center_x = x0 + TILE_W // 2
        badge_center_y = y1 - 18
        if font_label:
            _draw_label_badge(draw, img_canvas, label, badge_center_x, badge_center_y, font_label, badge_bg, badge_text, padding=8, min_width=70)

    return img_canvas


def get_sample_basket_images() -> list[dict]:
    """Get 12 sample basket images from the _static/images folder."""
    images_dir = os.path.join(project_root, "_static", "images")
    
    available = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith(".png") and f[0].isdigit()
    ])
    
    if len(available) < 12:
        print(f"Warning: Only found {len(available)} images, need 12")
        while len(available) < 12:
            available.extend(available[:12 - len(available)])
    
    selected = available[:12]
    return [{"image": f"images/{img}", "position": i + 1} for i, img in enumerate(selected)]


def main():
    print("=" * 60)
    print("AI Director Grid Composite - Test Generator")
    print("=" * 60)
    
    # Get sample baskets
    shared_grid = get_sample_basket_images()
    print(f"\nUsing {len(shared_grid)} basket images:")
    for slot in shared_grid:
        print(f"  Slot {slot['position']}: {slot['image']}")
    
    print("\nGenerating director grid composite...")
    
    # Generate the composite
    img = build_ai_director_grid_composite(shared_grid)
    
    # Save
    debug_dir = os.path.join(project_root, "_static", "ai_debug")
    os.makedirs(debug_dir, exist_ok=True)
    output_path = os.path.join(debug_dir, "ai_director_grid_TEST.png")
    
    img.save(output_path, format="PNG")
    
    print(f"\n✓ Saved test image to: {output_path}")
    print(f"  Image size: {img.size[0]}x{img.size[1]} pixels")
    
    # Open the image (macOS)
    if sys.platform == "darwin":
        print("\nOpening image...")
        os.system(f"open '{output_path}'")
    else:
        print(f"\nOpen the image at: {output_path}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
