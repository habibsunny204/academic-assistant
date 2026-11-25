"""Regression tests for the ingestion pipeline.

These pin down the behaviours that were wrong at least once during development
and that fail silently rather than loudly -- the dangerous kind for a system
whose whole purpose is being trustworthy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diu.chunk import chunk_document
from diu.clean import clean_text, is_noise
from diu.config import load_config
from diu.extract.meta import clean_filename_title, derive_metadata
from diu.extract.tables import RoutineState, clean_cell, table_to_blocks
from diu.models import DocMeta
from diu.store import tokenize


# -- cell normalisation ------------------------------------------------------

def test_clean_cell_restores_displaced_underscore():
    # PyMuPDF emits "CSE228(68_D)" as "CSE228(68 D) _".
    assert clean_cell("CSE228(68 D) _") == "CSE228(68_D)"
    assert clean_cell("EEE 3 _") == "EEE_3"
    assert clean_cell("NT-1") == "NT-1"


# -- routine table -----------------------------------------------------------

ROUTINE_HEADER = [
    ["SATURDAY", "", "", "", "", ""],
    ["08:30-10:00", "", "", "10:00-11:30", "", ""],
    ["Room", "Course", "Teacher", "Room", "Course", "Teacher"],
]


def test_routine_row_becomes_one_fact_per_class():
    rows = ROUTINE_HEADER + [
        ["KT-201", "CSE228(68 D) _", "MIS", "KT-201", "CSE315(67 B) _", "AS"],
    ]
    blocks = table_to_blocks(rows, page_no=1, state=RoutineState())
    assert len(blocks) == 2
    assert blocks[0].text == (
        "Course CSE228 section 68_D is scheduled on Saturday from 08:30-10:00 "
        "in room KT-201 taught by MIS."
    )
    assert blocks[1].meta["course_code"] == "CSE315"
    assert blocks[1].meta["day"] == "Saturday"


def test_routine_state_carries_day_across_pages():
    """A day's grid continues onto the next page with no headers at all."""
    state = RoutineState()
    table_to_blocks(ROUTINE_HEADER, page_no=1, state=state)

    continuation = [["KT-208", "PHY101(72 B) _", "MAK", "KT-208", "", ""]]
    blocks = table_to_blocks(continuation, page_no=2, state=state)

    assert len(blocks) == 1
    assert blocks[0].meta["day"] == "Saturday"
    assert blocks[0].meta["time_slot"] == "08:30-10:00"


def test_misaligned_column_is_dropped_not_guessed():
    """If the course column holds a room code, emit nothing rather than a wrong fact."""
    state = RoutineState()
    table_to_blocks(ROUTINE_HEADER, page_no=1, state=state)

    bad = [["KT-201", "KT-999", "MIS", "KT-201", "", ""]]
    blocks = table_to_blocks(bad, page_no=1, state=state)

    # No class fact is invented. The rows survive only as a verbatim Markdown
    # table, which is a faithful record rather than an asserted schedule.
    assert not any(b.kind == "table_fact" for b in blocks)
    assert all(b.meta.get("table_type") == "routine_aux" for b in blocks)
    assert state.rejected_rows == 1


def test_repeating_header_never_produces_row_sentences():
    """Pairing rows against a repeating Room/Course/Teacher header is nonsense."""
    rows = [
        ["Room", "Course", "Teacher", "Room", "Course", "Teacher"],
        ["KT-201", "CSE113(72 B) _", "SRH", "KT-201", "CSE221(69 F) _", "SZ"],
    ]
    blocks = table_to_blocks(rows, page_no=1, state=None)
    assert not any(b.meta.get("table_type") == "generic_row" for b in blocks)


# -- metadata ----------------------------------------------------------------

def test_filename_hash_suffix_is_stripped_from_title():
    assert clean_filename_title(
        "cse-class-routine-fall-2026-v11-1d709bde62.pdf"
    ) == "CSE Class Routine Fall 2026 V11"


def test_letterhead_is_not_used_as_a_title():
    """A notice whose first line is the university name gets its filename title."""
    meta = derive_metadata(
        "notice-for-late-registration-for-fall-2026-524065d2ca-474c4d51ed.pdf",
        "Daffodil International University (DIU)\nDaffodil Smart City (DSC), "
        "Birulia, Savar, Dhaka 1216\nNOTICE\nDate: 12 September 2026\n"
        "Students of Daffodil International University (DIU) are",
    )
    assert meta["title"] == "Notice for Late Registration for Fall 2026"


def test_dates_and_semester_are_extracted():
    meta = derive_metadata(
        "notice-for-late-registration-for-fall-2026.pdf",
        "NOTICE\nDate: 12 September 2026\nCourse registration for Fall 2026",
    )
    assert meta["semester"] == "Fall 2026"
    assert meta["issued_date"].isoformat() == "2026-09-12"
    assert meta["doc_type"] == "notice"
    assert meta["category"] == "Registration"


def test_routine_gets_its_own_category_not_curriculum():
    meta = derive_metadata("cse-class-routine-fall-2026-v11.pdf",
                           "Class Routine for CSE Program\nVersion V1.1")
    assert meta["doc_type"] == "routine"
    assert meta["category"] == "Class Routine"
    assert meta["version_label"] == "V1.1"


# -- cleaning ----------------------------------------------------------------

def test_undecodable_glyphs_are_removed():
    assert "�" not in clean_text("Dhaka � 1216, Bangladesh")


def test_page_numbers_are_noise():
    assert is_noise("- 12 -")
    assert is_noise("Page 4")
    assert not is_noise("The last date is 17 September 2026.")


# -- tokenisation ------------------------------------------------------------

def test_section_labels_survive_tokenisation():
    """"67_B" must stay one rare token, not split into "67" and "b"."""
    assert "67_b" in tokenize("Course CSE315 section 67_B")


# -- chunking ----------------------------------------------------------------

def test_every_chunk_carries_its_context_header():
    cfg = load_config()
    meta = DocMeta(
        doc_id="d_test", source_path="x.pdf", filename="x.pdf", sha256="0",
        title="Class Routine for CSE Program", semester="Fall 2026",
        version_label="V1.1",
    )
    rows = ROUTINE_HEADER + [
        ["KT-201", "CSE228(68 D) _", "MIS", "KT-201", "CSE315(67 B) _", "AS"],
    ]
    blocks = table_to_blocks(rows, page_no=1, state=RoutineState())
    chunks = chunk_document(blocks, meta, cfg)

    assert chunks
    for chunk in chunks:
        # Without this header the fact has no evidence it belongs to the CSE
        # routine for Fall 2026, and neither query term would retrieve it.
        assert "Class Routine for CSE Program" in chunk.text
        assert "Fall 2026" in chunk.text
        assert chunk.meta["semester"] == "Fall 2026"


def test_citation_payload_has_what_the_ui_renders():
    cfg = load_config()
    meta = DocMeta(
        doc_id="d_test", source_path="x.pdf", filename="x.pdf", sha256="0",
        title="Examination Policy", category="Examination",
    )
    from diu.models import Block

    blocks = [Block(kind="para", text="A student may apply for improvement. " * 8,
                    page_no=33)]
    chunk = chunk_document(blocks, meta, cfg)[0]
    citation = chunk.citation()
    assert citation["title"] == "Examination Policy"
    assert citation["page"] == 33
    assert citation["category"] == "Examination"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
