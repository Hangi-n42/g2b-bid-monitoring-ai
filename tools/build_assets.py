"""Build only static retrieval assets from competition-provided dev records.

No model training. The complete bank is permitted for final inference; local
evaluation excludes both the current ID/content and an explicit heldout fold.
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

ABSENCE = {10,11,16,18,20}
META_FIELDS = ('적용계약법','업무구분','계약방법','낙찰방법','입찰추정가격','배정예산금액','정보화사업여부')


def build_example(rec, label):
    """Expose only provided positive labels with retained direct evidence.

    Missing partial-label keys mean unannotated in this demonstration, never 0.
    A short excerpt cannot establish absence or certify all other items negative.
    No evidence is generated for positive labels whose supplied evidence is blank.
    """
    partial, evidence, spans = {}, {}, []
    for i in range(1,25):
        key, ev = f'v{i}', label.get(f'e{i}', '')
        if label[key] != '1' or i in ABSENCE or not ev:
            continue
        if len(ev)>500:
            raise ValueError(f'{rec["id"]} {key}: supplied evidence exceeds 500 characters')
        match = next(((idx,d['text'].find(ev)) for idx,d in enumerate(rec['docs']) if ev in d['text']), None)
        if match is None:
            raise ValueError(f'{rec["id"]} {key}: supplied evidence is not in source')
        doc, start = match
        partial[key], evidence[key] = 1, ev
        spans.append({'item':key,'doc':doc,'start':start,'end':start+len(ev),'text':ev})
    windows=[]
    for span in sorted(spans,key=lambda s:(s['doc'],s['start'])):
        text=rec['docs'][span['doc']]['text']
        start,end=max(0,span['start']-200),min(len(text),span['end']+200)
        if windows and windows[-1]['doc']==span['doc'] and start<=windows[-1]['end']:
            windows[-1]['end']=max(windows[-1]['end'],end)
        else:
            windows.append({'doc':span['doc'],'start':start,'end':end})
    chunks=[]
    for window in windows:
        d=rec['docs'][window['doc']]
        chunks.append(f'[{d["type"]}]\n'+d['text'][window['start']:window['end']])
    context='\n\n'.join(chunks)
    if any(ev not in context for ev in evidence.values()):
        raise ValueError('Evidence was lost while building example context')
    return {'schema_version':2,'id':rec['id'],'fingerprint':fingerprint(rec),
            'meta':{k:rec['meta'].get(k) for k in META_FIELDS},'context':context,
            'partial_labels':partial,'evidence':evidence,'evidence_spans':spans}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    with gzip.open(ROOT/'open/dev.jsonl.gz','rt',encoding='utf-8') as f:
        records=[json.loads(line) for line in f if line.strip()]
    with (ROOT/'open/dev_labels.csv').open(encoding='utf-8-sig',newline='') as f:
        labels={r['id']:r for r in csv.DictReader(f)}
    if set(labels)!={r['id'] for r in records}:
        raise ValueError('Dev labels and records have different IDs')
    bank=[build_example(rec,labels[rec['id']]) for rec in records]
    assets=ROOT/'submission/model'
    assets.mkdir(parents=True,exist_ok=True)
    (assets/'examples.json').write_text(json.dumps(bank,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    sources=[ROOT/'open/dev.jsonl.gz',ROOT/'open/dev_labels.csv',ROOT/'open/data/항목표.json']
    sources+=list((ROOT/'open/data/법령패키지').rglob('*'))
    manifest={'purpose':'static legal guides and dev search examples, no model weights',
        'example_count':len(bank),'external_data':False,'example_schema_version':2,
        'retrievable_example_count':sum(bool(r['partial_labels']) for r in bank),
        'provided_evidence_label_count':sum(len(r['partial_labels']) for r in bank),
        'example_policy':'Only supplied positive labels with exact retained evidence are demonstrated. Missing partial_labels keys are unknown, not negative. No absence labels or invented evidence.',
        'sources':{str(p.relative_to(ROOT)):sha(p) for p in sources if p.is_file()},
        'development_caveat':'Guides inspected all dev. Fold-excluded retrieval scores are internal diagnostics, not untouched validation.'}
    (assets/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    folds_path=ROOT/'analysis/dev_folds/manifest.json'
    if folds_path.exists():
        folds=json.loads(folds_path.read_text(encoding='utf-8'))
        for fold in folds['folds']:
            out=folds_path.parent/f'fold_{fold["fold"]}_exclude_ids.json'
            out.write_text(json.dumps(fold['excluded_retrieval_ids'],ensure_ascii=False),encoding='utf-8')
    print(f'Built {len(bank)} static dev examples and provenance manifest')

if __name__=='__main__': main()
