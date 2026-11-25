"""Turning tables into retrievable text.

A class routine is a relational table. Flattened to prose it becomes token soup
("KT-201 KT-201 CSE228 MIS KT-201 CSE315 AS...") that no embedding model can
match against "where is CSE315 on Saturday?". So each cell is expanded into a
self-contained fact sentence carrying every dimension of the table: day, time,
room, course, section and teacher.

The routine grid spans page boundaries: the day name, the time-slot row and the
Room/Course/Teacher header appear on the page where a day starts, and the
remaining rooms continue on the next page with no headers at all. RoutineState
carries that context forward, otherwise every continuation page silently
degrades into meaningless generic rows.

Generic tables fall back to one sentence per row plus a Markdown rendering.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import Block

WEEKDAYS = {
    "SATURDAY", "SUNDAY", "MONDAY", "TUESDAY",
    "WEDNESDAY", "THURSDAY", "FRIDAY",
}
TIME_RE = re.compile(r"^\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2}$")
COURSE_RE = re.compile(r"^([A-Z]{2,4}\s*\d{3})\s*\(\s*([^)]*?)\s*\)$")
# A course code is 2-4 letters immediately followed by 3 digits. Room codes
# ("KT-201", "G1-026") do not match, which is what makes this usable as a
# guard against mis-aligned columns.
COURSE_LIKE_RE = re.compile(r"\b[A-Z]{2,4}\s*\d{3}\b")


@dataclass
class RoutineState:
    """Routine context carried across tables and pages within one document."""

    day: str | None = None
    slots: dict[int, str] = field(default_factory=dict)
    triples: list[tuple[int, int, int]] = field(default_factory=list)
    seen_header: bool = False
    rejected_rows: int = 0


def clean_cell(value: str | None) -> str:
    """Normalise a table cell.

    PyMuPDF renders the underscore in "CSE228(68_D)" detached, yielding
    "CSE228(68 D) _". The trailing " _" is the displaced underscore, so put it
    back where it belongs rather than dropping the section label.
    """
    text = (value or "").replace("\n", " ").strip()
    while text.endswith(" _"):
        text = text[:-2].rstrip()
        if " " in text:
            head, _, tail = text.rpartition(" ")
            text = f"{head}_{tail}"
        else:
            text = f"{text}_"
    return re.sub(r"\s+", " ", text).strip()


def _is_day_row(cells: list[str]) -> str | None:
    filled = [c for c in cells if c]
    if len(filled) == 1 and filled[0].upper() in WEEKDAYS:
        return filled[0].upper().capitalize()
    return None


def _slot_columns(cells: list[str]) -> dict[int, str]:
    return {i: c for i, c in enumerate(cells) if TIME_RE.match(c)}


def _is_rct_header(cells: list[str]) -> bool:
    lowered = [c.lower() for c in cells]
    return lowered.count("room") >= 2 and lowered.count("course") >= 2


def _read_triples(cells: list[str]) -> list[tuple[int, int, int]]:
    triples = []
    for i, cell in enumerate(cells):
        if cell.lower() == "room" and i + 2 < len(cells):
            if cells[i + 1].lower() == "course" and cells[i + 2].lower() == "teacher":
                triples.append((i, i + 1, i + 2))
    return triples


def parse_routine(
    rows: list[list[str]], page_no: int, state: RoutineState
) -> list[Block]:
    """Expand a Room/Course/Teacher routine grid into one fact per class session.

    `state` is read and updated in place so that continuation pages inherit the
    day, time slots and column layout from the page that introduced them.
    """
    blocks: list[Block] = []

    for raw_row in rows:
        cells = [clean_cell(c) for c in raw_row]

        found_day = _is_day_row(cells)
        if found_day:
            state.day = found_day
            continue

        found_slots = _slot_columns(cells)
        if found_slots:
            state.slots = found_slots
            continue

        if _is_rct_header(cells):
            state.triples = _read_triples(cells)
            state.seen_header = True
            continue

        if not state.triples:
            continue

        for room_i, course_i, teacher_i in state.triples:
            if course_i >= len(cells):
                continue
            room, course, teacher = (
                cells[room_i], cells[course_i], cells[teacher_i]
            )
            if not course:
                continue  # empty slot: no class scheduled

            # If the column mapping has drifted, the "course" cell holds
            # something that is not a course code. Drop the row rather than
            # emit a confidently wrong fact.
            if not COURSE_LIKE_RE.search(course):
                state.rejected_rows += 1
                continue

            time_slot = state.slots.get(room_i, "")
            match = COURSE_RE.match(course)
            if match:
                code = match.group(1).replace(" ", "")
                section = match.group(2)
            else:
                code = COURSE_LIKE_RE.search(course).group(0).replace(" ", "")
                section = ""

            parts = [f"Course {code}"]
            if section:
                parts.append(f"section {section}")
            if state.day:
                parts.append(f"is scheduled on {state.day}")
            if time_slot:
                parts.append(f"from {time_slot}")
            if room:
                parts.append(f"in room {room}")
            if teacher:
                parts.append(f"taught by {teacher}")
            sentence = " ".join(parts) + "."

            blocks.append(
                Block(
                    kind="table_fact",
                    text=sentence,
                    page_no=page_no,
                    meta={
                        "table_type": "class_routine",
                        "course_code": code,
                        "section": section,
                        "day": state.day or "",
                        "time_slot": time_slot,
                        "room": room,
                        "teacher": teacher,
                    },
                )
            )
    return blocks


def _to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    padded = [list(r) + [""] * (width - len(r)) for r in rows]
    header, *body = padded
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def parse_generic_table(
    rows: list[list[str]], page_no: int, caption: str = ""
) -> list[Block]:
    """Keep the Markdown table for context, plus a sentence per row for recall."""
    cleaned = [[clean_cell(c) for c in row] for row in rows]
    cleaned = [r for r in cleaned if any(r)]
    if not cleaned:
        return []

    blocks: list[Block] = []
    markdown = _to_markdown(cleaned)
    if markdown:
        prefix = f"{caption}\n" if caption else ""
        blocks.append(
            Block(kind="table", text=prefix + markdown, page_no=page_no,
                  meta={"table_type": "generic", "n_rows": len(cleaned)})
        )

    header = cleaned[0]
    # Row sentences only make sense when the first row is a set of distinct
    # labels. A header like ["Room","Course","Teacher","Room","Course",...]
    # repeats, and pairing against it produces nonsense.
    labels = [c for c in header if c]
    header_is_labels = (
        len(labels) >= 2
        and len(set(labels)) == len(labels)
        and not any(c.isdigit() for c in labels)
    )
    if header_is_labels and len(cleaned) > 1:
        for row in cleaned[1:]:
            pairs = [
                f"{h}: {v}"
                for h, v in zip(header, row)
                if h and v and h.lower() != v.lower()
            ]
            if len(pairs) >= 2:
                text = "; ".join(pairs) + "."
                if caption:
                    text = f"{caption} - {text}"
                blocks.append(
                    Block(kind="table_fact", text=text, page_no=page_no,
                          meta={"table_type": "generic_row"})
                )
    return blocks


def table_to_blocks(
    rows: list[list[str]],
    page_no: int,
    caption: str = "",
    state: RoutineState | None = None,
) -> list[Block]:
    """Route a table to the routine specialiser, or fall back to generic."""
    cleaned = [[clean_cell(c) for c in row] for row in rows]
    is_routine = any(_is_rct_header(r) for r in cleaned) or (
        state is not None and state.seen_header
    )

    if is_routine and state is not None:
        facts = parse_routine(cleaned, page_no, state)
        if facts:
            return facts
        # A table inside a routine document that yields no class facts is
        # something else (a room legend, a footnote). Keep it as a table, but
        # never as generic row sentences: pairing against a repeating
        # Room/Course/Teacher header produces garbage.
        markdown = _to_markdown([r for r in cleaned if any(r)])
        if markdown:
            return [Block(kind="table", text=markdown, page_no=page_no,
                          meta={"table_type": "routine_aux"})]
        return []

    return parse_generic_table(rows, page_no, caption)
