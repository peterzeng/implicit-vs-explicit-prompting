from __future__ import annotations

import base64
import io
import json
import os
import random
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .prompt_context import _is_instruction_message
from .state import Player

# Static image cache for VLM context images
_STATIC_IMAGE_CACHE: dict[str, str] = {}

def _resolve_static_image_path(rel_path: str) -> str | None:
    """Resolve a static image path like 'images/025.png' to a filesystem path.

    We support a few layouts:
    - <project_root>/_static/<rel_path>
    - <project_root>/main/Human-VLM-Game/_static/<rel_path> (Heroku-style bundle)
    - An explicit STATIC_IMAGE_ROOT env override.
    """
    if not rel_path:
        return None

    rel_path = rel_path.lstrip("/ ")

    # Highest priority: explicit override
    explicit_root = os.environ.get("STATIC_IMAGE_ROOT", "").strip()
    candidates: list[str] = []
    if explicit_root:
        candidates.append(os.path.join(explicit_root, rel_path))

    # Project-root-based fallbacks
    app_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(app_dir)
    candidates.append(os.path.join(project_root, "_static", rel_path))
    candidates.append(
        os.path.join(project_root, "main", "Human-VLM-Game", "_static", rel_path)
    )

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _image_rel_to_data_url(rel_path: str) -> str | None:
    """Convert a static image path (e.g., 'images/025.png') to a data URL for GPT‑4o."""
    if not rel_path:
        return None

    # Simple in-memory cache to avoid re-reading the same files
    cache_key = rel_path.lstrip("/ ")
    cached = _STATIC_IMAGE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    fs_path = _resolve_static_image_path(cache_key)
    if not fs_path:
        return None

    try:
        with open(fs_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"
        _STATIC_IMAGE_CACHE[cache_key] = data_url
        return data_url
    except Exception:
        return None


def _load_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TrueType font with fallback to default.

    Tries common system font paths for a clean sans-serif font.
    """
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
    # Fallback to default
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _get_text_dimensions(
    draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont
) -> tuple[int, int]:
    """Get text dimensions using modern Pillow API with fallback."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # Fallback for older Pillow versions
        try:
            return draw.textsize(text, font=font)
        except Exception:
            return (len(text) * 8, 14)


def _draw_label_badge(
    draw: ImageDraw.Draw,
    img_canvas: Image.Image,
    text: str,
    center_x: int,
    center_y: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    bg_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
    padding: int = 6,
    min_width: int = 28,
) -> None:
    """Draw a text label with a rounded rectangle background badge."""
    tw, th = _get_text_dimensions(draw, text, font)
    badge_w = max(tw + padding * 2, min_width)
    badge_h = th + padding * 2

    x0 = center_x - badge_w // 2
    y0 = center_y - badge_h // 2
    x1 = x0 + badge_w
    y1 = y0 + badge_h

    # Draw rounded rectangle background
    radius = min(8, badge_h // 2)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=bg_color)

    # Draw text centered in badge
    tx = x0 + (badge_w - tw) // 2
    ty = y0 + (badge_h - th) // 2
    draw.text((tx, ty), text, font=font, fill=text_color)


def _draw_dashed_rect(
    draw: ImageDraw.Draw,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    dash_length: int = 10,
    gap_length: int = 6,
    width: int = 2,
) -> None:
    """Draw a dashed rectangle border."""
    # Top edge
    x = x0
    while x < x1:
        end_x = min(x + dash_length, x1)
        draw.line([(x, y0), (end_x, y0)], fill=color, width=width)
        x += dash_length + gap_length

    # Bottom edge
    x = x0
    while x < x1:
        end_x = min(x + dash_length, x1)
        draw.line([(x, y1), (end_x, y1)], fill=color, width=width)
        x += dash_length + gap_length

    # Left edge
    y = y0
    while y < y1:
        end_y = min(y + dash_length, y1)
        draw.line([(x0, y), (x0, end_y)], fill=color, width=width)
        y += dash_length + gap_length

    # Right edge
    y = y0
    while y < y1:
        end_y = min(y + dash_length, y1)
        draw.line([(x1, y), (x1, end_y)], fill=color, width=width)
        y += dash_length + gap_length


def _build_ai_director_grid_composite(player: Player) -> str | None:
    """
    Render a 2×6 grid image showing the 12 target baskets the AI director must describe.

    Each position (1–12) is drawn as a tile with the basket image and a clear slot label.
    This provides the director with a single composite image rather than 12 separate images,
    mirroring the visual context approach used for the matcher.

    Layout:
    - 2 rows × 6 columns for the 12 logical positions
    - Top row: Slots 1-6 (left to right)
    - Bottom row: Slots 7-12 (left to right)
    - Clear slot badges indicating the order for describing
    """
    if not hasattr(player, "group"):
        return None

    # Load the 12 target baskets from the shared grid
    try:
        shared_grid = json.loads(getattr(player.group, "shared_grid", "") or "[]")
    except Exception:
        shared_grid = []

    if not shared_grid:
        return None

    # Grid geometry: 2 rows × 6 columns for 12 baskets
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

    # Color scheme - consistent with matcher grid
    bg_color = (240, 242, 245)
    slot_bg = (255, 255, 255)
    border_color = (70, 130, 180)  # Steel blue for all slots
    header_color = (30, 40, 50)
    badge_bg = (70, 130, 180)  # Steel blue badges
    badge_text = (255, 255, 255)
    instruction_color = (80, 90, 100)

    img_canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(img_canvas)

    # Load fonts
    font_header = _load_font(24)
    font_label = _load_font(18)
    font_instruction = _load_font(14)

    # Get round number for header
    round_num = getattr(player, "round_number", 1) or 1

    # --- Header with ROUND NUMBER to distinguish from feedback images ---
    heading = f"ROUND {round_num} TARGET SEQUENCE (Baskets 1–12)"
    if font_header is not None:
        try:
            draw.text(
                (PADDING + 4, PADDING + 10),
                heading,
                font=font_header,
                fill=header_color,
            )
        except Exception:
            pass

    # --- Instruction line ---
    instruction = f"ROUND {round_num}: Describe in order: top row (1->6) then bottom row (7->12). Focus on one basket at a time."
    if font_instruction is not None:
        try:
            draw.text(
                (PADDING + 4, PADDING + HEADER_H + 4),
                instruction,
                font=font_instruction,
                fill=instruction_color,
            )
        except Exception:
            pass

    # --- 2×6 Grid of target baskets ---
    grid_origin_y = PADDING + HEADER_H + INSTRUCTION_H
    for logical_pos in range(1, 13):
        row = (logical_pos - 1) // COLS
        col = (logical_pos - 1) % COLS

        x0 = PADDING + col * (TILE_W + PADDING)
        y0 = grid_origin_y + row * (TILE_H + PADDING)
        x1 = x0 + TILE_W
        y1 = y0 + TILE_H

        # Draw slot background with border
        draw.rectangle([x0, y0, x1, y1], fill=slot_bg, outline=border_color, width=3)

        # Get the basket image for this position
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
                            # Fit basket image into the tile with margins
                            margin = 14
                            target_w = max(1, TILE_W - 2 * margin)
                            target_h = max(1, TILE_H - 2 * margin - 30)  # room for label
                            basket_img.thumbnail(
                                (target_w, target_h), Image.Resampling.LANCZOS
                            )
                            bw, bh = basket_img.size
                            bx = x0 + (TILE_W - bw) // 2
                            by = y0 + (TILE_H - bh) // 2 - 12
                            img_canvas.paste(basket_img, (bx, by))
                    except Exception:
                        pass

        # Draw position label badge at bottom center
        label = str(logical_pos)
        badge_center_x = x0 + TILE_W // 2
        badge_center_y = y1 - 18
        if font_label is not None:
            _draw_label_badge(
                draw,
                img_canvas,
                label,
                badge_center_x,
                badge_center_y,
                font_label,
                badge_bg,
                badge_text,
                padding=8,
                min_width=36,
            )

    # Save debug image
    try:
        _debug_save_ai_director_grid_image(player, img_canvas)
    except Exception:
        pass

    # Encode as data URL
    try:
        buf = io.BytesIO()
        img_canvas.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def _debug_save_ai_director_grid_image(player: Player, img_canvas: Image.Image) -> None:
    """
    Save the AI director grid composite as a PNG under the project's _static folder.

    This lets researchers visually confirm the image context sent to the director AI.
    """
    try:
        app_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(app_dir)
        debug_dir = os.path.join(project_root, "_static", "ai_debug")
        os.makedirs(debug_dir, exist_ok=True)

        session_code = ""
        try:
            if hasattr(player, "session") and player.session:
                session_code = getattr(player.session, "code", "") or ""
        except Exception:
            session_code = ""

        group_id = ""
        try:
            if hasattr(player, "group") and player.group:
                group_id = str(getattr(player.group, "id", "") or "")
        except Exception:
            group_id = ""

        round_num = ""
        try:
            round_num = str(getattr(player, "round_number", "") or "")
        except Exception:
            round_num = ""

        parts = ["ai_director_grid"]
        if session_code:
            parts.append(session_code)
        if group_id:
            parts.append(f"g{group_id}")
        if round_num:
            parts.append(f"r{round_num}")
        filename = "_".join(parts) + ".png"

        path = os.path.join(debug_dir, filename)
        img_canvas.save(path, format="PNG")
    except Exception:
        pass


def _build_ai_matcher_grid_composite(player: Player) -> str | None:
    """
    Render a 2×6 grid image showing the AI matcher's current 12-slot sequence.

    Each logical position (1–12) is drawn as a tile. If the AI has chosen a
    basket for that position, we render the basket image; otherwise the tile is
    shown as an empty placeholder with a dashed border. This mirrors the visual
    layout used in the researcher debug popup so that the model's visual
    context is 1:1 with what the human sees.

    Visual enhancements:
    - Large, readable fonts for all labels
    - Badge-style labels with background for high contrast
    - Dashed borders for empty slots
    - Prominent section headers
    - Legend explaining the blue border meaning
    """
    # Only meaningful when there is a group object.
    if not hasattr(player, "group"):
        return None

    # Load the incremental AI partial sequence accumulated so far.
    try:
        partial = json.loads(getattr(player.group, "ai_partial_sequence", "") or "[]")
    except Exception:
        partial = []

    # Build an index of image paths by explicit logical position 1–12.
    MAX_SLOTS = 12
    slot_images: list[str | None] = [None] * MAX_SLOTS
    if isinstance(partial, list):
        for item in partial:
            if not isinstance(item, dict):
                continue
            pos = item.get("position")
            img = (item.get("image") or "").lstrip("/ ")
            try:
                pos_int = int(pos)
            except Exception:
                continue
            if 1 <= pos_int <= MAX_SLOTS and img:
                # Later entries overwrite earlier ones for the same position.
                slot_images[pos_int - 1] = img

    # Load the matcher pool (targets + distractors) for the staging grid.
    try:
        pool_items = _load_matcher_pool_image_urls(player)
    except Exception:
        pool_items = []
    pool_paths: list[str | None] = []
    for item in pool_items or []:
        slot = item.get("slot") or {}
        img_path = (slot.get("image") or "").lstrip("/ ")
        if img_path:
            pool_paths.append(img_path)
    # Limit to the first 18 items (3×6 grid) just like the human staging area.
    MAX_POOL_SLOTS = 18
    if len(pool_paths) > MAX_POOL_SLOTS:
        pool_paths = pool_paths[:MAX_POOL_SLOTS]
    if len(pool_paths) < MAX_POOL_SLOTS:
        pool_paths.extend([None] * (MAX_POOL_SLOTS - len(pool_paths)))

    # Track which pool baskets are already used in the 12-slot sequence so we
    # can lightly highlight them in the staging grid.
    used_paths = {p for p in slot_images if p}

    # Grid geometry:
    #   - Top: 2 rows × 6 columns for the 12 logical positions (target row)
    #   - Bottom: 3 rows × 6 columns for the staging pool of candidates
    COLS = 6
    TARGET_ROWS = 2
    STAGING_ROWS = 3
    TILE_W = 220
    TILE_H = 220
    PADDING = 12
    HEADER_H = 40  # increased for larger headers
    LEGEND_H = 32  # space for legend at bottom
    canvas_w = COLS * TILE_W + (COLS + 1) * PADDING

    target_height = TARGET_ROWS * TILE_H + (TARGET_ROWS + 1) * PADDING
    staging_height = STAGING_ROWS * TILE_H + (STAGING_ROWS + 1) * PADDING
    canvas_h = (
        PADDING
        + HEADER_H
        + target_height
        + PADDING
        + HEADER_H
        + staging_height
        + PADDING
        + LEGEND_H
        + PADDING
    )

    # Color scheme - refined for better contrast
    bg_color = (240, 242, 245)  # slightly darker background
    slot_bg_empty = (255, 255, 255)  # white for empty slots
    slot_bg_filled = (255, 255, 255)  # white for filled slots
    border_color = (180, 186, 194)
    border_empty = (160, 170, 180)  # slightly different for empty
    border_selected = (41, 128, 185)  # stronger blue for selected
    text_color = (50, 60, 70)
    header_color = (30, 40, 50)
    badge_bg_slot = (70, 130, 180)  # steel blue for slot badges
    badge_bg_candidate = (100, 100, 110)  # gray for candidate badges
    badge_text = (255, 255, 255)  # white text on badges
    empty_text_color = (150, 160, 170)  # muted for empty slot placeholder

    img_canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(img_canvas)

    # Load fonts at different sizes
    font_header = _load_font(22)
    font_label = _load_font(18)
    font_small = _load_font(14)
    font_empty = _load_font(32)  # large "?" for empty slots

    # Get round number for header
    round_num = getattr(player, "round_number", 1) or 1

    # --- Section Header: Current Sequence with ROUND NUMBER ---
    heading1 = f"ROUND {round_num} - YOUR CURRENT SEQUENCE (Positions 1–12)"
    if font_header is not None:
        try:
            draw.text(
                (PADDING + 4, PADDING + 8),
                heading1,
                font=font_header,
                fill=header_color,
            )
        except Exception:
            pass

    # --- Top block: 12-slot target grid (2×6) ---
    target_origin_y = PADDING + HEADER_H
    for logical_pos in range(1, MAX_SLOTS + 1):
        row = (logical_pos - 1) // COLS
        col = (logical_pos - 1) % COLS

        x0 = PADDING + col * (TILE_W + PADDING)
        y0 = target_origin_y + row * (TILE_H + PADDING)
        x1 = x0 + TILE_W
        y1 = y0 + TILE_H

        rel_img_path = slot_images[logical_pos - 1]
        is_empty = rel_img_path is None

        if is_empty:
            # Draw empty slot with dashed border
            draw.rectangle([x0, y0, x1, y1], fill=slot_bg_empty)
            _draw_dashed_rect(draw, x0, y0, x1, y1, border_empty, dash_length=12, gap_length=8, width=3)

            # Draw large "?" in center for empty slots
            if font_empty is not None:
                try:
                    qw, qh = _get_text_dimensions(draw, "?", font_empty)
                    qx = x0 + (TILE_W - qw) // 2
                    qy = y0 + (TILE_H - qh) // 2 - 15
                    draw.text((qx, qy), "?", font=font_empty, fill=empty_text_color)
                except Exception:
                    pass
        else:
            # Draw filled slot with solid border
            draw.rectangle([x0, y0, x1, y1], fill=slot_bg_filled, outline=border_color, width=3)

            fs_path = _resolve_static_image_path(rel_img_path)
            if fs_path and os.path.exists(fs_path):
                try:
                    with Image.open(fs_path) as basket_img:
                        basket_img = basket_img.convert("RGB")
                        # Fit basket image into the tile with margins
                        margin = 14
                        target_w = max(1, TILE_W - 2 * margin)
                        target_h = max(1, TILE_H - 2 * margin - 30)  # leave room for label
                        basket_img.thumbnail(
                            (target_w, target_h), Image.Resampling.LANCZOS
                        )
                        bw, bh = basket_img.size
                        bx = x0 + (TILE_W - bw) // 2
                        by = y0 + (TILE_H - bh) // 2 - 12
                        img_canvas.paste(basket_img, (bx, by))
                except Exception:
                    pass

        # Draw position label badge at bottom center
        label = str(logical_pos)
        badge_center_x = x0 + TILE_W // 2
        badge_center_y = y1 - 18
        if font_label is not None:
            _draw_label_badge(
                draw,
                img_canvas,
                label,
                badge_center_x,
                badge_center_y,
                font_label,
                badge_bg_slot,
                badge_text,
                padding=8,
                min_width=36,
            )

    # --- Section Header: Candidate Pool with ROUND NUMBER ---
    staging_origin_y = target_origin_y + target_height + PADDING + HEADER_H
    heading2 = f"ROUND {round_num} - CANDIDATE POOL (Choose from these baskets)"
    if font_header is not None:
        try:
            draw.text(
                (PADDING + 4, staging_origin_y - HEADER_H + 8),
                heading2,
                font=font_header,
                fill=header_color,
            )
        except Exception:
            pass

    # --- Bottom block: 3×6 staging grid of candidate baskets ---
    for idx, rel_path in enumerate(pool_paths):
        row = idx // COLS
        col = idx % COLS

        x0 = PADDING + col * (TILE_W + PADDING)
        y0 = staging_origin_y + row * (TILE_H + PADDING)
        x1 = x0 + TILE_W
        y1 = y0 + TILE_H

        is_used = rel_path and rel_path in used_paths
        outline_color = border_selected if is_used else border_color
        outline_width = 4 if is_used else 2

        draw.rectangle([x0, y0, x1, y1], fill=slot_bg_filled, outline=outline_color, width=outline_width)

        if rel_path:
            fs_path = _resolve_static_image_path(rel_path)
            if fs_path and os.path.exists(fs_path):
                try:
                    with Image.open(fs_path) as basket_img:
                        basket_img = basket_img.convert("RGB")
                        margin = 14
                        target_w = max(1, TILE_W - 2 * margin)
                        target_h = max(1, TILE_H - 2 * margin - 30)  # leave room for label
                        basket_img.thumbnail(
                            (target_w, target_h), Image.Resampling.LANCZOS
                        )
                        bw, bh = basket_img.size
                        bx = x0 + (TILE_W - bw) // 2
                        by = y0 + (TILE_H - bh) // 2 - 12
                        img_canvas.paste(basket_img, (bx, by))
                except Exception:
                    pass

        # Draw candidate index badge at bottom center
        label = str(idx + 1)
        badge_center_x = x0 + TILE_W // 2
        badge_center_y = y1 - 18
        if font_label is not None:
            _draw_label_badge(
                draw,
                img_canvas,
                label,
                badge_center_x,
                badge_center_y,
                font_label,
                badge_bg_candidate,
                badge_text,
                padding=8,
                min_width=36,
            )

    # --- Legend at bottom ---
    legend_y = canvas_h - LEGEND_H - PADDING + 4
    if font_small is not None:
        try:
            # Draw legend items
            legend_x = PADDING + 8

            # Blue border indicator
            draw.rectangle(
                [legend_x, legend_y + 4, legend_x + 24, legend_y + 24],
                fill=None,
                outline=border_selected,
                width=3,
            )
            legend_x += 32
            draw.text(
                (legend_x, legend_y + 6),
                "= Already placed in sequence",
                font=font_small,
                fill=text_color,
            )

            # Dashed border indicator
            legend_x += 220
            _draw_dashed_rect(
                draw,
                legend_x,
                legend_y + 4,
                legend_x + 24,
                legend_y + 24,
                border_empty,
                dash_length=6,
                gap_length=4,
                width=2,
            )
            legend_x += 32
            draw.text(
                (legend_x, legend_y + 6),
                "= Empty position (needs a basket)",
                font=font_small,
                fill=text_color,
            )
        except Exception:
            pass

    # Persist a debug PNG under _static/ so researchers can visually confirm
    # what the model sees. This intentionally mirrors the AI matcher debug row.
    try:
        _debug_save_ai_matcher_grid_image(player, img_canvas)
    except Exception:
        # Debugging is best-effort only; never break the main flow.
        pass

    # Encode the composite as a data URL for GPT‑4o.
    try:
        buf = io.BytesIO()
        img_canvas.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def _build_round_feedback_image(
    player: Player, viewer_role: str | None = None
) -> str | None:
    """
    Render a feedback image showing round results with correct/incorrect highlighting.

    The image is role-aware:
    - Matcher feedback shows the submitted basket in each slot.
    - Director feedback shows the correct target basket in each slot, with red
      borders only marking the positions the matcher got wrong. This preserves
      the director's information boundary: the director learns which positions
      were misunderstood, but not which basket the matcher actually selected.

    This mirrors what the players should see on the RoundFeedback page:
    - A 2×6 grid of the 12 positions
    - Green border for correct placements
    - Red border for incorrect placements
    - Header showing "Round X Feedback: Y/12 correct"

    Returns a data URL for the image, or None if generation fails.
    """
    if not hasattr(player, "group"):
        return None

    # Load the correct sequence and matcher's submissions
    try:
        shared_grid = json.loads(player.group.shared_grid or "[]")
        matcher_sequence = json.loads(player.group.matcher_sequence or "[]")
    except Exception:
        return None

    if not shared_grid:
        return None

    correct_count, slots_data = _prepare_round_feedback_slots(
        shared_grid, matcher_sequence, viewer_role=viewer_role
    )

    if slots_data is None:
        return None

    # Grid geometry
    COLS = 6
    ROWS = 2
    TILE_W = 180
    TILE_H = 180
    PADDING = 10
    HEADER_H = 50
    LEGEND_H = 30

    canvas_w = COLS * TILE_W + (COLS + 1) * PADDING
    grid_height = ROWS * TILE_H + (ROWS + 1) * PADDING
    canvas_h = PADDING + HEADER_H + grid_height + LEGEND_H + PADDING

    # Colors
    bg_color = (245, 247, 250)
    correct_border = (40, 167, 69)  # Green
    incorrect_border = (220, 53, 69)  # Red
    correct_bg = (234, 247, 238)  # Light green
    incorrect_bg = (248, 215, 218)  # Light red
    text_color = (50, 60, 70)
    badge_text = (255, 255, 255)

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None

    img_canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(img_canvas)

    # Load fonts
    try:
        header_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except Exception:
        try:
            header_font = ImageFont.truetype("arial.ttf", 28)
        except Exception:
            header_font = ImageFont.load_default()

    try:
        badge_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except Exception:
        try:
            badge_font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            badge_font = ImageFont.load_default()

    # Draw header
    round_num = getattr(player, "round_number", "?")
    header_text = f"Round {round_num} Feedback: {correct_count}/12 correct"
    header_y = PADDING + 10
    try:
        bbox = draw.textbbox((0, 0), header_text, font=header_font)
        text_w = bbox[2] - bbox[0]
    except Exception:
        text_w = len(header_text) * 12
    header_x = (canvas_w - text_w) // 2
    draw.text((header_x, header_y), header_text, fill=text_color, font=header_font)

    # Draw grid
    grid_top = PADDING + HEADER_H
    for idx, slot in enumerate(slots_data):
        row = idx // COLS
        col = idx % COLS
        x = PADDING + col * (TILE_W + PADDING)
        y = grid_top + PADDING + row * (TILE_H + PADDING)

        # Choose colors based on correctness
        if slot["is_correct"]:
            border = correct_border
            fill = correct_bg
        else:
            border = incorrect_border
            fill = incorrect_bg

        # Draw slot background
        draw.rectangle([x, y, x + TILE_W, y + TILE_H], fill=fill, outline=border, width=4)

        # Draw basket image if present
        img_path = slot["image"]
        if img_path:
            try:
                full_path = _resolve_static_image_path(img_path)
                if full_path and os.path.isfile(full_path):
                    basket_img = Image.open(full_path).convert("RGBA")
                    # Scale to fit with padding
                    inner_size = min(TILE_W, TILE_H) - 20
                    basket_img.thumbnail((inner_size, inner_size), Image.Resampling.LANCZOS)
                    # Center in tile
                    bx = x + (TILE_W - basket_img.width) // 2
                    by = y + (TILE_H - basket_img.height) // 2
                    img_canvas.paste(basket_img, (bx, by), basket_img)
            except Exception:
                pass

        # Draw position badge
        badge_text_str = str(slot["position"])
        badge_w, badge_h = 28, 24
        badge_x = x + 6
        badge_y = y + 6
        badge_color = correct_border if slot["is_correct"] else incorrect_border
        draw.rounded_rectangle(
            [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
            radius=6,
            fill=badge_color,
        )
        try:
            bbox = draw.textbbox((0, 0), badge_text_str, font=badge_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = 10, 14
        draw.text(
            (badge_x + (badge_w - tw) // 2, badge_y + (badge_h - th) // 2 - 1),
            badge_text_str,
            fill=badge_text,
            font=badge_font,
        )

    # Draw legend
    legend_y = grid_top + grid_height + 5
    legend_items = [
        (correct_border, "Correct"),
        (incorrect_border, "Incorrect"),
    ]
    legend_x = PADDING + 10
    for color, label in legend_items:
        draw.rectangle([legend_x, legend_y, legend_x + 16, legend_y + 16], fill=color)
        draw.text((legend_x + 22, legend_y - 2), label, fill=text_color, font=badge_font)
        legend_x += 100

    # Save debug copy locally
    _debug_save_round_feedback_image(player, img_canvas, viewer_role=viewer_role)

    # Encode as data URL
    try:
        buf = io.BytesIO()
        img_canvas.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def _prepare_round_feedback_slots(
    shared_grid: list[dict[str, Any]],
    matcher_sequence: list[dict[str, Any]],
    viewer_role: str | None = None,
) -> tuple[int, list[dict[str, Any]] | None]:
    """Return correctness count and role-appropriate feedback slot payloads."""
    # Build correct sequence
    correct_sequence = [slot.get("image") for slot in shared_grid]

    # Build matcher's submissions by position
    matcher_by_pos = {}
    for item in matcher_sequence or []:
        if not isinstance(item, dict):
            continue
        pos = item.get("position")
        try:
            pos_int = int(pos)
        except (TypeError, ValueError):
            continue
        if 1 <= pos_int <= 12 and pos_int not in matcher_by_pos:
            matcher_by_pos[pos_int] = item

    # Count correct placements and prepare slot data
    correct_count = 0
    slots_data = []
    for i in range(12):
        correct_img = correct_sequence[i] if i < len(correct_sequence) else None
        submitted_entry = matcher_by_pos.get(i + 1)
        submitted_img = submitted_entry.get("image") if submitted_entry else None

        is_correct = (
            submitted_img is not None
            and correct_img is not None
            and submitted_img == correct_img
        )
        if is_correct:
            correct_count += 1

        display_img = correct_img if viewer_role == "director" else submitted_img

        slots_data.append({
            "position": i + 1,
            "image": display_img,
            "is_correct": is_correct,
        })
    return correct_count, slots_data


def _inject_round_feedback_context(
    player: Player, messages: list[dict[str, Any]], ai_role: str | None = None
) -> list[dict[str, Any]]:
    """Insert prior-round submitted-grid feedback images into the prompt.

    These images are only historical context. They should stay near the
    conversation from the same round so the model does not see a stack of
    visually similar grids detached from the dialogue they explain.
    """
    try:
        current_round = int(getattr(player, "round_number", 1) or 1)
    except Exception:
        current_round = 1
    if current_round <= 1:
        return messages

    try:
        all_round_players = player.in_all_rounds()
    except Exception:
        all_round_players = [player]

    feedback_messages_by_round: dict[int, dict[str, Any]] = {}
    for p_round in all_round_players:
        try:
            round_num = int(getattr(p_round, "round_number", 0) or 0)
        except Exception:
            continue
        if not (1 <= round_num < current_round):
            continue

        feedback_url = _build_round_feedback_image(p_round, viewer_role=ai_role)
        if not feedback_url:
            continue

        if ai_role == "director":
            feedback_text = (
                f"*** ROUND {round_num} DIRECTOR FEEDBACK (PAST ROUND) ***\n"
                "This historical image shows the correct target basket for each 12-position slot "
                "from a previous round. Green means the matcher placed that position correctly; "
                "red means the matcher got that position wrong. Red slots show the correct basket "
                "for that position, not the basket the matcher selected. The same physical baskets "
                "recur across rounds, so use this to learn which descriptions were misunderstood "
                "and which basket identities need clearer labels. Do NOT reuse old position numbers "
                "for the current round."
            )
        else:
            feedback_text = (
                f"*** ROUND {round_num} SUBMITTED GRID FEEDBACK (PAST ROUND) ***\n"
                "This historical image shows the MATCHER's submitted 12-position grid "
                "from a previous round. Green means that submitted position was correct; "
                "red means that exact submitted basket was incorrect for that described target. "
                "The same physical baskets recur across rounds, so use this to recover shared labels, "
                "visual conventions, correct identities, and prior wrong guesses. Do NOT reuse its "
                "old position numbers or old candidate numbers for the current round."
            )

        feedback_messages_by_round[round_num] = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": feedback_text,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": feedback_url},
                },
            ],
        }

    if not feedback_messages_by_round:
        return messages

    grouped_messages: list[dict[str, Any]] = []
    inserted_rounds: set[int] = set()

    for msg in messages:
        grouped_messages.append(msg)
        content = msg.get("content")
        if not isinstance(content, str):
            continue

        for round_num, feedback_message in feedback_messages_by_round.items():
            if round_num in inserted_rounds:
                continue
            marker = f"═══ ROUND {round_num} HISTORY "
            if content.startswith(marker):
                grouped_messages.append(feedback_message)
                inserted_rounds.add(round_num)
                break

    if len(inserted_rounds) == len(feedback_messages_by_round):
        return grouped_messages

    # Fallback for unusual prompt strategies with no explicit round markers:
    # keep instructions first, then put any ungrouped historical images before
    # the remaining conversation.
    insert_idx = 0
    while insert_idx < len(grouped_messages) and _is_instruction_message(grouped_messages[insert_idx]):
        insert_idx += 1
    missing_feedback = [
        feedback_messages_by_round[round_num]
        for round_num in sorted(feedback_messages_by_round)
        if round_num not in inserted_rounds
    ]
    return grouped_messages[:insert_idx] + missing_feedback + grouped_messages[insert_idx:]


def _debug_save_round_feedback_image(
    player: Player, img_canvas: Image.Image, viewer_role: str | None = None
) -> None:
    """
    Save the round feedback image as a PNG under the project's _static folder.

    This lets researchers review the visual feedback that would be shown to the AI
    for cross-round history context.
    """
    try:
        app_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(app_dir)
        debug_dir = os.path.join(project_root, "_static", "ai_debug")
        os.makedirs(debug_dir, exist_ok=True)

        session_code = ""
        try:
            if hasattr(player, "session") and player.session:
                session_code = getattr(player.session, "code", "") or ""
        except Exception:
            session_code = ""

        group_id = ""
        try:
            if hasattr(player, "group") and player.group:
                group_id = str(getattr(player.group, "id", "") or "")
        except Exception:
            group_id = ""

        round_num = ""
        try:
            round_num = str(getattr(player, "round_number", "") or "")
        except Exception:
            round_num = ""

        parts = ["round_feedback"]
        if viewer_role in ("director", "matcher"):
            parts.append(viewer_role)
        if session_code:
            parts.append(session_code)
        if group_id:
            parts.append(f"g{group_id}")
        if round_num:
            parts.append(f"r{round_num}")
        filename = "_".join(parts) + ".png"

        path = os.path.join(debug_dir, filename)
        img_canvas.save(path, format="PNG")
    except Exception:
        # Strictly best-effort; ignore any filesystem errors.
        pass


def _debug_save_ai_matcher_grid_image(player: Player, img_canvas: Image.Image) -> None:
    """
    Save the AI matcher grid composite as a PNG under the project's _static folder.

    This lets researchers open a static URL or file and visually confirm that the
    image context sent to GPT‑4o matches the AI matcher debug view and human UI.
    """
    try:
        app_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(app_dir)
        debug_dir = os.path.join(project_root, "_static", "ai_debug")
        os.makedirs(debug_dir, exist_ok=True)

        session_code = ""
        try:
            if hasattr(player, "session") and player.session:
                session_code = getattr(player.session, "code", "") or ""
        except Exception:
            session_code = ""

        group_id = ""
        try:
            if hasattr(player, "group") and player.group:
                group_id = str(getattr(player.group, "id", "") or "")
        except Exception:
            group_id = ""

        round_num = ""
        try:
            round_num = str(getattr(player, "round_number", "") or "")
        except Exception:
            round_num = ""

        parts = ["ai_matcher_grid"]
        if session_code:
            parts.append(session_code)
        if group_id:
            parts.append(f"g{group_id}")
        if round_num:
            parts.append(f"r{round_num}")
        filename = "_".join(parts) + ".png"

        path = os.path.join(debug_dir, filename)
        img_canvas.save(path, format="PNG")
    except Exception:
        # Strictly best-effort; ignore any filesystem errors.
        return None


def _load_shared_grid_image_urls(player: Player) -> list[dict[str, Any]]:
    """Return a list of {'slot': slot_dict, 'data_url': 'data:image/...'} for the shared grid.

    If images cannot be resolved, returns an empty list and callers should gracefully
    fall back to text-only prompting.
    """
    if not hasattr(player, "group"):
        return []
    try:
        shared_grid = json.loads(getattr(player.group, "shared_grid", "") or "[]")
    except Exception:
        shared_grid = []

    results: list[dict[str, Any]] = []
    for slot in shared_grid or []:
        img_path = (slot.get("image") or "").lstrip("/ ")
        if not img_path:
            continue
        data_url = _image_rel_to_data_url(img_path)
        if not data_url:
            continue
        results.append({"slot": slot, "data_url": data_url})
    return results


def _load_matcher_pool_image_urls(player: Player) -> list[dict[str, Any]]:
    """Return image URLs for the matcher's full choice pool (targets + distractors).

    For the MATCHER role we want to approximate the staging area the human
    matcher sees: the 12 target baskets from the director's grid plus the
    additional distractor baskets drawn from the preset `fullList` for the
    configured basket set.
    """
    # Start with the 12 target baskets from the shared grid
    base = _load_shared_grid_image_urls(player)
    if not base:
        return base

    # Build a set of already-included relative image paths
    seen_paths: set[str] = set()
    for item in base:
        slot = item.get("slot") or {}
        img_path = (slot.get("image") or "").lstrip("/ ")
        if img_path:
            seen_paths.add(img_path)

    # Load preset fullList for this basket_set, mirroring DraggableGridPage.vars_for_template
    try:
        if hasattr(player, "session") and player.session:
            try:
                set_num = int(player.session.config.get("basket_set", 1))
            except Exception:
                set_num = 1
        else:
            set_num = 1
        if set_num == 2:
            preset_filename = "grids_presets2.json"
        elif set_num == 3:
            preset_filename = "grids_presets3.json"
        elif set_num == 4:
            preset_filename = "grids_presets4.json"
        elif set_num == 5:
            preset_filename = "grids_presets5.json"
        else:
            preset_filename = "grids_presets1.json"
        preset_path = os.path.join(os.path.dirname(__file__), preset_filename)
        with open(preset_path, "r", encoding="utf-8") as f:
            presets = json.load(f)
        preset_full_list: list[str] = []
        for item in presets.get("rounds", []):
            if isinstance(item, dict) and "fullList" in item:
                preset_full_list = [
                    f"images/{img}" for img in item.get("fullList", []) or []
                ]
                break
    except Exception:
        preset_full_list = []

    # Add a small number of extras from preset_full_list that aren't already in
    # the 12‑basket grid.
    extras: list[dict[str, Any]] = []
    MAX_EXTRAS = 6
    for rel_path in preset_full_list or []:
        if len(extras) >= MAX_EXTRAS:
            break
        rel_path = rel_path.lstrip("/ ")
        if rel_path in seen_paths:
            continue
        data_url = _image_rel_to_data_url(rel_path)
        if not data_url:
            continue
        extras.append(
            {
                "slot": {"image": rel_path, "basket_id": None},
                "data_url": data_url,
            }
        )

    # Use the same deterministic shuffle as DraggableGridPage.vars_for_template
    # so the AI matcher sees the exact same basket arrangement as the human matcher.
    combined = base + extras
    try:
        import random
        round_num = int(getattr(player, "round_number", 1) or 1)
        seed = 4242 + (set_num * 100) + round_num
        rng = random.Random(seed)
        rng.shuffle(combined)
    except Exception:
        pass  # If shuffle fails, return unshuffled
    return combined


def _inject_visual_grid_context(
    player: Player, messages: list[dict[str, Any]], ai_role: str | None = None
):
    """Inject a multimodal grid message so the AI sees the 12-basket layout.

    This wrapper is applied on top of all prompt strategies.
    so that the only differences between strategies are in how the model is
    instructed to reason and respond, not in whether it has visual access
    to the baskets.
    """
    if not messages:
        return messages

    if ai_role not in ("director", "matcher"):
        human_role = (
            player.field_maybe_none("player_role") or player.participant.vars.get("role")
        )
        ai_role = "matcher" if human_role == "director" else "director"
    if ai_role not in ("director", "matcher"):
        return messages

    # Both roles now receive a single composite grid image:
    # - Director: 2×6 grid of the 12 target baskets to describe
    # - Matcher: 12-slot sequence (top) + candidate pool (bottom)
    composite_url = None
    if ai_role == "director":
        composite_url = _build_ai_director_grid_composite(player)
    else:
        composite_url = _build_ai_matcher_grid_composite(player)

    if not composite_url:
        # Shapes demo or static images missing; fall back to text-only prompts.
        import logging
        logging.warning("[VISUAL_CONTEXT] No composite_url generated for %s", ai_role)
        return messages
    
    import logging
    # logging.info(
        # "[VISUAL_CONTEXT] Generated composite image for %s, URL length: %d bytes",
        # ai_role, len(composite_url) if composite_url else 0
    # )

    # Get current round number for explicit context
    current_round = getattr(player, "round_number", 1) or 1

    if ai_role == "director":
        intro_text = (
            f"*** ROUND {current_round} TARGET GRID ***\n"
            f"This image (labeled 'ROUND {current_round} TARGET SEQUENCE') shows the 12 baskets you must describe for THIS round.\n\n"
            f"CRITICAL: The same physical basket set appears across rounds, but Round {current_round} is in a DIFFERENT order. "
            f"Carry forward useful names and corrections, but do not reuse previous position numbers. "
            f"ONLY describe the baskets in THIS image, labeled 'ROUND {current_round} TARGET SEQUENCE'.\n\n"
            "Layout: 2 rows × 6 columns with Baskets 1–6 on the top row and Baskets 7–12 on the bottom row. "
            "IMPORTANT: Describe ONE BASKET PER MESSAGE, in order. Wait for your partner to confirm before moving to the next basket."
        )
    else:
        intro_text = (
            f"*** ROUND {current_round} MATCHER VIEW ***\n"
            f"This image shows your current sequence state for THIS round.\n\n"
            f"CRITICAL: The same physical basket set appears across rounds, but Round {current_round} has different candidate numbers and sequence positions. "
            f"Carry forward useful names, correct matches, and wrong-match feedback, but map them onto THIS current candidate pool. "
            f"ONLY select from the candidates shown in THIS image.\n\n"
            "Layout: TOP TWO ROWS show your CURRENT 12-position sequence (positions 1–12). "
            "BOTTOM THREE ROWS show your CANDIDATE POOL of 18 baskets to choose from. "
            "Match the DIRECTOR's descriptions to candidates in THIS image only."
        )

    multimodal_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": intro_text,
        },
        {
            "type": "image_url",
            "image_url": {
                "url": composite_url,
            },
        },
    ]

    grid_message = {
        "role": "user",
        "content": multimodal_content,
    }

    # Insert after historical round context, but before current-round active
    # dialogue when that boundary is present. This keeps old feedback images
    # paired with old transcripts and makes the current grid the freshest
    # visual frame for the model's next action.
    idx = 0
    while idx < len(messages) and _is_instruction_message(messages[idx]):
        idx += 1

    current_marker = f"═══ ROUND {current_round} (CURRENT ROUND - ACTIVE) ═══"
    for current_idx in range(idx, len(messages)):
        content = messages[current_idx].get("content")
        if isinstance(content, str) and content.startswith(current_marker):
            return messages[:current_idx] + [grid_message] + messages[current_idx:]

    saw_past_history = any(
        isinstance(message.get("content"), str)
        and " HISTORY (PAST - SAME BASKETS, DIFFERENT ORDER) ═══" in message.get("content", "")
        for message in messages[idx:]
    )
    if saw_past_history:
        return messages + [grid_message]

    return messages[:idx] + [grid_message] + messages[idx:]
