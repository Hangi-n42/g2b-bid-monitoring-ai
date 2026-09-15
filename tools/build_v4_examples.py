"""Compile agent-reviewed excerpts from provided dev; never create new labels.

Review observations are methodology notes, not legal sources or CSV evidence.
Each row demonstrates exactly one existing dev label. Negative evidence is a
counterexample excerpt in a prompt, never a nonempty eN in submission.csv.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'submission'))
from script import fingerprint

# Reviewed against the supplied record text and dev_labels.csv on 2026-09-15.
# Tuple: record, item, existing label, document, exact anchor, review observation.
# Review selections are local data and are intentionally not published.


def build_reviewed(records,labels, specs=None):
    if specs is None:
        path=ROOT/'submission/model/review_specs.json'
        if not path.exists():
            raise FileNotFoundError('Restore local review_specs.json; review data is not published.')
        specs=json.loads(path.read_text(encoding='utf-8'))
    fields=('적용계약법','업무구분','계약방법','낙찰방법','입찰추정가격','배정예산금액','정보화사업여부')
    out=[]
    for rid,item,value,doc,anchor,why in specs:
        rec=records[rid];key=f'v{item}'
        if int(labels[rid][key])!=value:
            raise ValueError(f'Review label does not match provided gold: {rid} {key}')
        text=rec['docs'][doc]['text'];start=text.find(anchor)
        if start<0 or text.count(anchor)!=1:
            raise ValueError(f'Expected unique exact reviewed anchor: {rid} {key}')
        end=start+len(anchor)
        # Extra following context retains the actual exception/eligibility
        # continuation reviewed for v15/16/17/20; no unrelated automatic fill.
        after=550 if item in {15,16,17} else 350 if item==20 else 140
        left=max(0,start-100);right=min(len(text),end+after)
        excerpt=text[left:right]
        out.append({'schema_version':3,'example_id':f'{rid}:{key}:reviewed',
            'id':rid,'source_id':rid,'fingerprint':fingerprint(rec),
            'meta':{k:rec['meta'].get(k) for k in fields},
            'context':f'[{rec["docs"][doc]["type"]}]\n'+excerpt,
            'partial_labels':{key:value},'evidence':{key:anchor},
            'evidence_spans':[{'item':key,'doc':doc,'start':start,'end':end,'text':anchor}],
            'context_spans':[{'doc':doc,'start':left,'end':right,'text':excerpt}],
            'review':{'reviewer':'agent','date':'2026-09-15','label_source':'open/dev_labels.csv',
                'text_source':'open/dev.jsonl.gz','label_changed':False,'why':why,
                'source_document_sha256':hashlib.sha256(text.encode('utf-8')).hexdigest(),
                'original_evidence_empty':not bool(labels[rid].get(f'e{item}')),
                'scope':'Only this provided item label; other items are unspecified. Counterexample excerpts are not submission CSV evidence.'}})
    return out


def main():
    with gzip.open(ROOT/'open/dev.jsonl.gz','rt',encoding='utf-8') as stream:
        records={r['id']:r for r in map(json.loads,stream)}
    with (ROOT/'open/dev_labels.csv').open(encoding='utf-8-sig',newline='') as stream:
        labels={r['id']:r for r in csv.DictReader(stream)}
    out=build_reviewed(records,labels)
    path=ROOT/'submission/model/reviewed_examples.json'
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Built {len(out)} agent-reviewed demonstrations; original dev labels unchanged')


if __name__=='__main__':main()
