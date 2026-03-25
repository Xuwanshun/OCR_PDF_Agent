from __future__ import annotations

import json
from typing import Any

from .schemas import DocumentJSON


def render_markdown(doc: DocumentJSON) -> str:
    lines: list[str] = []
    lines.append(f"# Document: {doc.source_path}")
    lines.append("")
    for page in doc.pages:
        lines.append(f"## Page {page.page_number}")
        lines.append("")
        for b in page.blocks:
            t = b.text.strip()
            if not t:
                continue
            lines.append(t)
            lines.append("")

        if page.regions:
            lines.append("### Regions")
            lines.append("")
            for r in page.regions:
                lines.append(f"- Region {r.region_id} ({r.type}) bbox={r.bbox}")
                if r.extraction:
                    lines.append("")
                    lines.append("```json")
                    lines.append(json.dumps(r.extraction, ensure_ascii=False, indent=2))
                    lines.append("```")
                    lines.append("")
    return "\n".join(lines).strip() + "\n"

