"""CJK-aware markdown table aligner for docs/API_GUIDE_zh.md.

Convention (matches the committed file): cell = "| " + content + padding + "|"
where content+padding spans width+1 columns; separator = "-" * width.
Escaped pipes (\\|) are preserved verbatim; widths count the unescaped glyph.
Idempotent. Usage: uv run python scripts/align_zh_tables.py
"""

import re
import sys
import unicodedata

PATH = "docs/API_GUIDE_zh.md"


def dwidth(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def is_sep(line: str) -> bool:
    return bool(re.match(r"^\|(\s*:?-+:?\s*\|)+$", line.strip()))


def parse(line: str) -> list[str]:
    return [c for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]


def align_lines(lines: list[str]) -> tuple[list[str], int, int]:
    fixed = tables = 0
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|") and i + 1 < len(lines) and is_sep(lines[i + 1]):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
            rows = [parse(ln) for ln in lines[i:j]]
            ncols = len(rows[0])
            if all(len(r) == ncols for r in rows):
                tables += 1
                cells = [[c.strip() for c in r] for r in rows]
                widths = [max(dwidth(c) for c in col) for col in zip(*cells)]

                def build(row: list[str], sep: bool) -> str:
                    if sep:
                        return indent + "|" + "|".join("-" * w for w in widths) + "|"
                    # Widths count the cell as written (escaped pipes included) —
                    # this matches the committed table convention byte-for-byte
                    parts = [c + " " * (widths[k] - dwidth(c)) for k, c in enumerate(row)]
                    return indent + "| " + " | ".join(parts) + " |"

                new = [build(r, sep=(k == 1)) for k, r in enumerate(cells)]
                if new != lines[i:j]:
                    fixed += 1
                lines[i:j] = new
            i = j
        else:
            i += 1
    return lines, tables, fixed


def main() -> int:
    with open(PATH, encoding="utf-8") as f:
        lines = f.read().split("\n")
    lines, tables, fixed = align_lines(lines)
    with open(PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    # idempotency check
    again, _, refixed = align_lines(list(lines))
    assert again == lines, "aligner not idempotent"
    print(f"tables={tables} repadded={fixed} idempotent=yes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
