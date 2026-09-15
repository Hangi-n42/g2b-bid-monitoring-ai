"""Source-retention invariants, using only the public competition records."""
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'submission'))
from context import select_segments


def public_records():
    with gzip.open(ROOT / 'open/dev.jsonl.gz', 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def test_full_budget_preserves_every_character_once():
    for rec in public_records():
        budget = sum(len(d['text']) for d in rec['docs'])
        spans = select_segments(rec, budget)
        assert sum(len(s['text']) for s in spans) == budget
        for doc_index, doc in enumerate(rec['docs']):
            parts = [s for s in spans if s['doc'] == doc_index]
            assert ''.join(s['text'] for s in parts) == doc['text']
            cursor = 0
            for part in parts:
                assert part['start'] == cursor
                assert 0 < len(part['text']) <= 360
                cursor = part['end']
            assert cursor == len(doc['text'])


def test_selection_is_bounded_exact_and_independent():
    recs = public_records()
    before = select_segments(recs[0], 16000)
    for rec in reversed(recs):
        for budget in (11000, 16000):
            spans = select_segments(rec, budget)
            assert sum(len(s['text']) for s in spans) <= budget
            assert [s['sid'] for s in spans] == list(range(1, len(spans) + 1))
            for s in spans:
                assert s['text'] == rec['docs'][s['doc']]['text'][s['start']:s['end']]
                assert 0 < len(s['text']) <= 360
    assert before == select_segments(recs[0], 16000)


def test_public_evidence_not_lost_at_revised_budget():
    # Check both exact selectable evidence and continuous source coverage. These
    # public reference snippets do not measure prediction or evidence accuracy.
    recs = {r['id']: r for r in public_records()}
    with (ROOT / 'open/dev_labels.csv').open(encoding='utf-8-sig', newline='') as stream:
        labels = list(csv.DictReader(stream))
    checked = 0
    for row in labels:
        rec = recs[row['id']]
        spans = select_segments(rec, 16000)
        for item in range(1, 25):
            evidence = row[f'e{item}']
            if row[f'v{item}'] != '1' or not evidence:
                continue
            checked += 1
            assert any(evidence in s['text'] for s in spans), (row['id'], item)
            found = False
            for doc_index, doc in enumerate(rec['docs']):
                pos = doc['text'].find(evidence)
                while pos >= 0:
                    cursor = pos
                    for s in spans:
                        if s['doc'] == doc_index and s['start'] <= cursor < s['end']:
                            cursor = s['end']
                    if cursor >= pos + len(evidence):
                        found = True
                        break
                    pos = doc['text'].find(evidence, pos + 1)
                if found:
                    break
            assert found, (row['id'], item)
    assert checked == 54
