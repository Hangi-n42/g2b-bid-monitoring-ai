"""Prevent partial excerpts from teaching unsupported complete label vectors."""
import csv
import gzip
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from build_assets import ABSENCE, build_example


@pytest.fixture(scope='module')
def source_data():
    with gzip.open(ROOT/'open/dev.jsonl.gz','rt',encoding='utf-8') as stream:
        records={r['id']:r for r in map(json.loads,stream)}
    with (ROOT/'open/dev_labels.csv').open(encoding='utf-8-sig',newline='') as stream:
        labels={r['id']:r for r in csv.DictReader(stream)}
    return records,labels


def test_every_demonstrated_label_has_its_own_supplied_evidence(source_data):
    records,labels=source_data
    demonstrated=0
    for rid,rec in records.items():
        example=build_example(rec,labels[rid])
        expected={f'v{i}' for i in range(1,25) if i not in ABSENCE
                  and labels[rid][f'v{i}']=='1' and labels[rid][f'e{i}']}
        assert 'labels' not in example
        assert set(example['partial_labels'])==set(example['evidence'])==expected
        assert set(example['partial_labels'].values()) <= {1}
        for span in example['evidence_spans']:
            ev=rec['docs'][span['doc']]['text'][span['start']:span['end']]
            assert ev==span['text']==example['evidence'][span['item']]
            assert ev in example['context']
        demonstrated+=len(expected)
    assert demonstrated==54  # Provided, nonempty positive evidence; no generated labels.


def test_known_incomplete_v19_cases_do_not_teach_unsupported_positive(source_data):
    records,labels=source_data
    for rid in ['PPS-DEV-034','PPS-DEV-035','PPS-DEV-036','PPS-DEV-037']:
        assert labels[rid]['v19']=='1' and labels[rid]['e19']==''
        example=build_example(records[rid],labels[rid])
        assert 'v19' not in example['partial_labels']
        assert not example['context']


def test_absence_and_negative_labels_never_inferred_from_excerpt(source_data):
    records,labels=source_data
    for rid,rec in records.items():
        example=build_example(rec,labels[rid])
        assert not set(example['partial_labels']) & {f'v{i}' for i in ABSENCE}
        if not any(labels[rid][f'v{i}']=='1' for i in range(1,25)):
            assert example['partial_labels']=={} and example['context']==''


def test_bad_supplied_evidence_fails_instead_of_creating_false_demonstration(source_data):
    records,labels=source_data
    rid='PPS-DEV-01'
    broken=dict(labels[rid])
    # Invalid offset/excerpt handling only; this is not a fabricated training label.
    broken['e1']='\x00'
    with pytest.raises(ValueError,match='not in source'):
        build_example(records[rid],broken)
