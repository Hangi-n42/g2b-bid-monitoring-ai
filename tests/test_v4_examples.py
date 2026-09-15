"""Checks for source-grounded reviewed examples; no model-score claims."""
import csv,gzip,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from build_v4_examples import build_reviewed


def test_reviewed_asset_has_existing_gold_and_exact_source_spans():
    with gzip.open(ROOT/'open/dev.jsonl.gz','rt',encoding='utf-8') as stream:
        records={r['id']:r for r in map(json.loads,stream)}
    with (ROOT/'open/dev_labels.csv').open(encoding='utf-8-sig',newline='') as stream:
        labels={r['id']:r for r in csv.DictReader(stream)}
    expected=build_reviewed(records,labels)
    actual=json.loads((ROOT/'submission/model/reviewed_examples.json').read_text(encoding='utf-8'))
    assert actual==expected
    assert len({r['example_id'] for r in actual})==len(actual)
    for row in actual:
        assert len(row['partial_labels'])==1
        key,value=next(iter(row['partial_labels'].items()))
        assert value==int(labels[row['id']][key])
        assert row['review']['label_changed'] is False
        assert row['review']['reviewer']=='agent'
        for span in row['evidence_spans']+row['context_spans']:
            text=records[row['id']]['docs'][span['doc']]['text']
            assert span['text']==text[span['start']:span['end']]
            assert span['text'] in row['context']
            assert hashlib.sha256(text.encode('utf-8')).hexdigest()==row['review']['source_document_sha256']
    negatives={k for r in actual for k,v in r['partial_labels'].items() if v==0}
    assert {'v10','v11','v16','v18','v20'} <= negatives
    positive=next(r for r in actual if r['partial_labels'].get('v19')==1)
    assert positive['partial_labels']=={'v19':1}
    assert positive['review']['original_evidence_empty']
    assert positive['evidence']['v19'] in positive['context']
    negative=next(r for r in actual if r['partial_labels'].get('v19')==0)
    assert negative['evidence']['v19'] in negative['context']
