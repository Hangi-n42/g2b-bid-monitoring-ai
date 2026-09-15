"""Group-local source invariants; supplied dev evidence is not an F1 test."""
import csv
import gzip
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'submission'))
from context import select_segments

GROUPS = (tuple(range(1, 9)), tuple(range(10, 19)), (9, *range(19, 25)))


@pytest.fixture(scope='module')
def records():
    with gzip.open(ROOT / 'open/dev.jsonl.gz', 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def test_all_groups_cover_exact_original_spans_and_local_ids(records):
    assert sorted(i for group in GROUPS for i in group) == list(range(1, 25))
    for rec in records:
        for group in GROUPS:
            spans = select_segments(rec, 6000, items=group)
            assert sum(len(s['text']) for s in spans) <= 6000
            assert [s['sid'] for s in spans] == list(range(1, len(spans) + 1))
            assert [(s['doc'], s['start']) for s in spans] == sorted((s['doc'], s['start']) for s in spans)
            for s in spans:
                assert 0 < len(s['text']) <= 360
                assert s['text'] == rec['docs'][s['doc']]['text'][s['start']:s['end']]


def test_public_reference_evidence_retained_in_responsible_group(records):
    indexed = {r['id']: r for r in records}
    with (ROOT / 'open/dev_labels.csv').open(encoding='utf-8-sig', newline='') as stream:
        labels = list(csv.DictReader(stream))
    checked = 0
    for row in labels:
        rec = indexed[row['id']]
        for group in GROUPS:
            spans = select_segments(rec, 6000, items=group)
            for i in group:
                evidence = row[f'e{i}']
                if row[f'v{i}'] == '1' and evidence:
                    checked += 1
                    assert any(evidence in s['text'] for s in spans), (row['id'], i)
    assert checked == 54


def test_group_selection_does_not_depend_on_other_notices_or_call_order(records):
    rec = records[0]
    before = select_segments(rec, 6000, items=GROUPS[2])
    select_segments(records[-1], 6000, items=GROUPS[0])
    select_segments(rec, 6000, items=GROUPS[1])
    assert before == select_segments(rec, 6000, items=tuple(f'v{i}' for i in GROUPS[2]))


@pytest.mark.parametrize('items', [(), (0,), (25,), ('v0',), ('bad',), (True,)])
def test_invalid_group_rejected(records, items):
    with pytest.raises(ValueError):
        select_segments(records[0], 6000, items=items)
