from __future__ import annotations

import html
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from referential_task.prompt_context import TASK_BACKGROUND
from referential_task.prompts import acl as ACL_prompt
from referential_task.prompts import cameron as cameron_module


OUT_DIR = Path("prompt_exports")


@dataclass
class PromptBlock:
    title: str
    kind: str
    content: str


class FakePlayer:
    def __init__(self, human_role: str, round_number: int = 1, num_rounds: int = 5):
        self.player_role = human_role
        self.round_number = round_number
        self.participant = SimpleNamespace(vars={"role": human_role})
        self.session = SimpleNamespace(config={"num_rounds": num_rounds})

    def field_maybe_none(self, name: str):
        return getattr(self, name, None)


def strategy_blocks(strategy: str, ai_role: str) -> list[PromptBlock]:
    human_role = "director" if ai_role == "matcher" else "matcher"
    player = FakePlayer(human_role)
    if strategy == "ACL_prompt":
        messages = ACL_prompt.build_acl_prompt_messages(player, None, [], ai_role=ai_role)
    elif strategy == "cameron-prompt":
        messages = cameron_module.build_cameron_prompt_messages(player, None, [], ai_role=ai_role)
    else:
        raise ValueError(strategy)

    return [
        PromptBlock(
            title=f"{strategy} / {ai_role.title()} / Developer Message {i + 1}",
            kind=message["role"],
            content=message["content"],
        )
        for i, message in enumerate(messages)
        if message.get("role") == "developer"
    ]


def visual_context_text(ai_role: str, style: str) -> str:
    current_round = "{current_round}"
    if ai_role == "director":
        return (
            f"ROUND {current_round} TARGET GRID: This image shows the 12 baskets you must describe for ROUND {current_round}.\n\n"
            "The grid shows 2 rows × 6 columns with Baskets 1–6 on the top row and Baskets 7–12 on the bottom row. "
            "IMPORTANT: Pair this image only with the Round "
            f"{current_round} conversation below. Describe ONE BASKET PER MESSAGE, in order (1, 2, 3, ..., 12). "
            "Wait for your partner to confirm before moving to the next basket. "
            "Your MATCHER partner sees these 12 baskets mixed with 6 additional distractors in their candidate pool."
        )
    if style == "natural":
        return (
            f"ROUND {current_round} CANDIDATE POOL: This image shows the 18 candidates you can choose from for ROUND {current_round}.\n\n"
            "The pool contains 12 TRUE TARGETS (which the DIRECTOR will describe) mixed with 6 DISTRACTORS. "
            "Each candidate is numbered 1-18. Pair this image only with the Round "
            f"{current_round} conversation below.\n\n"
            "When you identify a basket, respond naturally and state which candidate number (1-18) you're "
            "placing in which position (1-12). For example: 'Got it! I'll place candidate 7 in position 3.'"
        )
    return (
        f"ROUND {current_round} CANDIDATE POOL: This image shows the 18 candidates you can choose from for ROUND {current_round}.\n\n"
        "The pool contains 12 TRUE TARGETS (which the DIRECTOR will describe) mixed with 6 DISTRACTORS. "
        "Each candidate is numbered 1-18. Pair this image only with the Round "
        f"{current_round} conversation below. Use these numbers in your action tags (e.g., [PLACE:7,3]).\n\n"
        "IMPORTANT: Look at this image to find the candidate that matches each description, then include "
        "the candidate NUMBER in your [PLACE:C,P] tag."
    )


def shared_runtime_blocks() -> list[PromptBlock]:
    return [
        PromptBlock(
            "Shared Runtime Wrapper / Task Background",
            "developer",
            TASK_BACKGROUND,
        ),
        PromptBlock(
            "Shared Runtime Wrapper / Director Visual Context Text",
            "user + image",
            visual_context_text("director", "ACL_prompt"),
        ),
        PromptBlock(
            "Shared Runtime Wrapper / Matcher Visual Context Text",
            "user + image",
            visual_context_text("matcher", "ACL_prompt"),
        ),
        PromptBlock(
            "Shared Runtime Wrapper / Director Start-of-Round Message",
            "user",
            (
                "START OF ROUND {current_round}: This is a NEW round with the baskets in a DIFFERENT ORDER. "
                "The basket positions have been reshuffled - Basket 1 in this round is NOT the same as Basket 1 from previous rounds. "
                "Please describe ONLY Basket 1 (top-left in the grid) for now. "
                "Do NOT describe multiple baskets - just Basket 1. Wait for my response before moving to Basket 2."
            ),
        ),
        PromptBlock(
            "Shared Runtime Wrapper / Matcher Sequence-State Message",
            "developer",
            (
                "AUTHORITATIVE CURRENT MATCHER SEQUENCE STATE (for this turn):\n"
                "- There are 12 positions total.\n"
                "- `sequence_candidate_indices` is a length-12 array aligned to positions 1..12.\n"
                "- A value of null means that position is EMPTY/unfilled right now.\n"
                "- Default `reasoning.target_position` is the LOWEST-NUMBERED null entry in `sequence_candidate_indices` (unless the DIRECTOR explicitly revisits a specific basket number).\n"
                "- You MUST NOT set `selection.ready_to_submit` true if ANY entry is null.\n"
                '{"sequence_candidate_indices": [null, null, null, null, null, null, null, null, null, null, null, null]}'
            ),
        ),
        PromptBlock(
            "Cameron Matcher Extra Runtime JSON Instruction",
            "developer",
            (
                'You MUST respond with valid JSON containing BOTH an "utterance" field AND a "selection" field:\n'
                "{\n"
                '  "utterance": "<your natural language response to show in the chat - describe what you see, ask questions, or confirm your choice>",\n'
                '  "selection": {\n'
                '    "candidate_index": <integer 1–18 from the numbered candidate tiles, or null if asking for clarification>,\n'
                '    "position": <integer 1–12 for which position this basket goes in, or null for next available>,\n'
                '    "ready_to_submit": <true only when submitting final 12‑basket order, otherwise false>\n'
                "  }\n"
                "}\n\n"
                "Rules:\n"
                '- The "utterance" field is REQUIRED - this is what the human will see in the chat.\n'
                "- Never mention candidate indices, IDs, or filenames in your utterance.\n"
                "- If you reuse a candidate_index already placed elsewhere, the system moves it (old position becomes empty).\n"
                "- Set ready_to_submit to true only once, when you're confident in all 12 positions."
            ),
        ),
    ]


def all_blocks() -> list[PromptBlock]:
    blocks = shared_runtime_blocks()
    for strategy in ("ACL_prompt", "cameron-prompt"):
        for ai_role in ("director", "matcher"):
            blocks.extend(strategy_blocks(strategy, ai_role))
    return blocks


def markdown_document(blocks: list[PromptBlock]) -> str:
    lines = [
        "# Basket Referential Game AI Prompt Reference",
        "",
        "Generated from the repository prompt builders. Dynamic runtime values are shown with braces, for example `{current_round}`. The visual context blocks are sent with an image in the actual API call; this document includes the text paired with those images.",
        "",
        "## Message Order Notes",
        "",
        "At runtime, the shared task background is prepended before the strategy-specific developer messages. Visual context is inserted after developer/system instructions. Matchers also receive the current sequence-state message. Cameron matchers receive an additional JSON-format instruction.",
        "",
    ]
    for block in blocks:
        lines.extend(
            [
                f"## {block.title}",
                "",
                f"Message role/type: `{block.kind}`",
                "",
                "```text",
                block.content,
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def html_document(blocks: list[PromptBlock]) -> str:
    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Basket Referential Game AI Prompt Reference</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;line-height:1.45;color:#111}h1{font-size:24pt}h2{font-size:16pt;margin-top:28px}p.meta{color:#555}pre{white-space:pre-wrap;font-family:Consolas,Menlo,monospace;font-size:10pt;background:#f7f7f7;border:1px solid #ddd;padding:12px;border-radius:6px}code{font-family:Consolas,Menlo,monospace}</style>",
        "</head><body>",
        "<h1>Basket Referential Game AI Prompt Reference</h1>",
        "<p>Generated from the repository prompt builders. Dynamic runtime values are shown with braces, for example <code>{current_round}</code>. The visual context blocks are sent with an image in the actual API call; this document includes the text paired with those images.</p>",
        "<h2>Message Order Notes</h2>",
        "<p class='meta'>At runtime, the shared task background is prepended before the strategy-specific developer messages. Visual context is inserted after developer/system instructions. Matchers also receive the current sequence-state message. Cameron matchers receive an additional JSON-format instruction.</p>",
    ]
    for block in blocks:
        body.append(f"<h2>{html.escape(block.title)}</h2>")
        body.append(f"<p class='meta'>Message role/type: <code>{html.escape(block.kind)}</code></p>")
        body.append(f"<pre>{html.escape(block.content)}</pre>")
    body.append("</body></html>")
    return "\n".join(body)


def docx_paragraph(text: str, style: str | None = None) -> str:
    p_style = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    runs = []
    for line_number, line in enumerate(text.split("\n")):
        if line_number:
            runs.append("<w:r><w:br/></w:r>")
        runs.append(f"<w:r><w:t xml:space=\"preserve\">{escape(line)}</w:t></w:r>")
    return f"<w:p>{p_style}{''.join(runs)}</w:p>"


def make_docx(path: Path, blocks: list[PromptBlock]) -> None:
    paragraphs = [
        docx_paragraph("Basket Referential Game AI Prompt Reference", "Title"),
        docx_paragraph(
            "Generated from the repository prompt builders. Dynamic runtime values are shown with braces, for example {current_round}. The visual context blocks are sent with an image in the actual API call; this document includes the text paired with those images."
        ),
        docx_paragraph("Message Order Notes", "Heading1"),
        docx_paragraph(
            "At runtime, the shared task background is prepended before the strategy-specific developer messages. Visual context is inserted after developer/system instructions. Matchers also receive the current sequence-state message. Cameron matchers receive an additional JSON-format instruction."
        ),
    ]
    for block in blocks:
        paragraphs.append(docx_paragraph(block.title, "Heading1"))
        paragraphs.append(docx_paragraph(f"Message role/type: {block.kind}", "Subtitle"))
        paragraphs.append(docx_paragraph(block.content, "Code"))

    document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {''.join(paragraphs)}
    <w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>
  </w:body>
</w:document>'''
    styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:sz w:val="22"/></w:rPr><w:pPr><w:spacing w:after="160" w:line="276" w:lineRule="auto"/></w:pPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="48"/></w:rPr><w:pPr><w:spacing w:after="200"/></w:pPr></w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/><w:rPr><w:color w:val="555555"/><w:sz w:val="20"/></w:rPr><w:pPr><w:spacing w:after="120"/></w:pPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr><w:pPr><w:keepNext/><w:spacing w:before="360" w:after="160"/></w:pPr></w:style>
  <w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:basedOn w:val="Normal"/><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="20"/></w:rPr><w:pPr><w:spacing w:after="160" w:line="276" w:lineRule="auto"/></w:pPr></w:style>
</w:styles>'''
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'''
    rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''
    doc_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'''

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/_rels/document.xml.rels", doc_rels)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/styles.xml", styles_xml)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    blocks = all_blocks()
    (OUT_DIR / "basket_ai_prompt_reference.md").write_text(
        markdown_document(blocks), encoding="utf-8"
    )
    (OUT_DIR / "basket_ai_prompt_reference.html").write_text(
        html_document(blocks), encoding="utf-8"
    )
    (OUT_DIR / "basket_ai_prompt_reference.json").write_text(
        json.dumps([block.__dict__ for block in blocks], indent=2), encoding="utf-8"
    )
    make_docx(OUT_DIR / "basket_ai_prompt_reference.docx", blocks)


if __name__ == "__main__":
    main()
