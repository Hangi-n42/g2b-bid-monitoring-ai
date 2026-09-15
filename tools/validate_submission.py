"""Strict local submission validation and optional macro-F1, standard library only.

Validation never modifies predictions. Evidence is checked against individual
document bodies, not meta or a synthetic join spanning multiple documents.
"""
from __future__ import annotations
import argparse, csv, gzip, io, json, unicodedata
from pathlib import Path

COLUMNS = ['id']+[f'v{i}' for i in range(1,25)]+[f'e{i}' for i in range(1,25)]
ABSENCE = {10,11,16,18,20}


def read_records(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate(path, records):
    errors, warnings = [], []
    raw = Path(path).read_bytes()
    if raw.startswith(b'\xef\xbb\xbf'): errors.append('UTF-8 BOM is forbidden')
    try: text = raw.decode('utf-8-sig')
    except UnicodeDecodeError as exc: return {'errors':[str(exc)], 'warnings':[]}, []
    if unicodedata.normalize('NFC', text) != text: errors.append('Output is not NFC normalized')
    try:
        reader = csv.DictReader(io.StringIO(text,newline=''), strict=True)
        if reader.fieldnames != COLUMNS: errors.append('Expected exact 49-column header id,v1..v24,e1..e24')
        rows = list(reader)
    except csv.Error as exc: return {'errors':[str(exc)], 'warnings':[]}, []
    expected = {r['id']:r for r in records}
    if len(expected) != len(records): errors.append('Input contains duplicate IDs')
    if len(rows) != len(records): errors.append(f'Row count {len(rows)} != input count {len(records)}')
    seen = set()
    for index,row in enumerate(rows,2):
        rid = row.get('id')
        tag = f'line {index}, id={rid}'
        if None in row or any(value is None for value in row.values()): errors.append(f'{tag}: wrong field count')
        if rid in seen: errors.append(f'{tag}: duplicate ID')
        seen.add(rid)
        if rid not in expected: errors.append(f'{tag}: unexpected ID'); continue
        texts = [d['text'] for d in expected[rid]['docs']]
        for i in range(1,25):
            v, ev = row.get(f'v{i}'), row.get(f'e{i}') or ''
            if v not in ('0','1'): errors.append(f'{tag}: v{i} must be integer literal 0 or 1')
            if len(ev)>500: errors.append(f'{tag}: e{i} exceeds 500 characters')
            if ev.startswith(('=','+','@')): errors.append(f'{tag}: e{i} forbidden formula prefix')
            if (v=='0' or i in ABSENCE) and ev: errors.append(f'{tag}: e{i} must be empty')
            if ev and not any(ev in text for text in texts): errors.append(f'{tag}: e{i} is not a document substring')
            if v=='1' and i not in ABSENCE and not ev: warnings.append(f'{tag}: positive v{i} lacks evidence for second-round review')
    missing = set(expected)-seen
    if missing: errors.append(f'Missing IDs: {sorted(missing)}')
    return {'rows':len(rows), 'errors':errors, 'warnings':warnings}, rows


def score(rows, labels_path):
    with open(labels_path,encoding='utf-8-sig',newline='') as stream: labels = {r['id']:r for r in csv.DictReader(stream)}
    predictions = {r['id']:r for r in rows}
    if set(labels)!=set(predictions): raise ValueError('Scoring requires exact equality of prediction and label ID sets; use an explicit split label file')
    items = {}
    for i in range(1,25):
        key = f'v{i}'
        tp = sum(labels[rid][key]=='1' and p[key]=='1' for rid,p in predictions.items())
        fp = sum(labels[rid][key]=='0' and p[key]=='1' for rid,p in predictions.items())
        fn = sum(labels[rid][key]=='1' and p[key]=='0' for rid,p in predictions.items())
        denom = 2*tp+fp+fn
        items[key] = {'tp':tp,'fp':fp,'fn':fn,'support':tp+fn,'f1':2*tp/denom if denom else 0.0}
    return {'macro_f1':sum(r['f1'] for r in items.values())/24, 'items':items, 'zero_support_items':[k for k,v in items.items() if not v['support']], 'zero_division':0}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('submission')
    parser.add_argument('--input',required=True)
    parser.add_argument('--labels')
    parser.add_argument('--report')
    args=parser.parse_args()
    result, rows = validate(args.submission,read_records(args.input))
    if args.labels and not result['errors']: result['score'] = score(rows,args.labels)
    rendered=json.dumps(result,ensure_ascii=False,indent=2)
    if args.report:
        Path(args.report).parent.mkdir(parents=True,exist_ok=True)
        Path(args.report).write_text(rendered,encoding='utf-8')
    print(rendered)
    raise SystemExit(bool(result['errors']))


if __name__=='__main__': main()
