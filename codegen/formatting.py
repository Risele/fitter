from __future__ import annotations

from typing import List


def emit_wrapped_array_items(items: List[str], indent: str = "  ", max_line_len: int = 80) -> List[str]:
    if max_line_len < len(indent) + 4:
        max_line_len = len(indent) + 4
    lines: List[str] = []
    current = indent
    for item in items:
        token = item + ","
        # maintain a single space between items on the same line
        candidate = token if current == indent else " " + token
        if current == indent:
            if len(current) + len(token) <= max_line_len:
                current += token
            else:
                lines.append(indent + token)
        else:
            if len(current) + len(candidate) <= max_line_len:
                current += candidate
            else:
                lines.append(current)
                if len(indent) + len(token) <= max_line_len:
                    current = indent + token
                else:
                    lines.append(indent + token)
                    current = indent
    if current != indent:
        lines.append(current)
    return lines
