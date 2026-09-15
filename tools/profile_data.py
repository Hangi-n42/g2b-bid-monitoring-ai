"""Profile only supplied contest inputs; no model or external data is used."""
from __future__ import annotations
import argparse, collections, csv, gzip, hashlib, json, math
from pathlib import Path


def records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def fingerprint(rec):
    # Identity independent of opaque record/document IDs.
    content = [(d['type'], d['text']) for d in rec['docs']]
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def summary(values):
    values = sorted(values)
    return {name: values[round((len(values)-1)*q)] for name, q in [('min',0),('p25',.25),('median',.5),('p75',.75),('p90',.9),('p95',.95),('max',1)]} if values else {}


def profile(path):
    lengths, docs, meta, complete, hashes, ids = [], collections.Counter(), collections.defaultdict(collections.Counter), collections.Counter(), {}, set()
    duplicate_ids, duplicates, missing, replacement = [], [], 0, 0
    for rec in records(path):
        rid = rec['id']
        if rid in ids: duplicate_ids.append(rid)
        ids.add(rid)
        fp = fingerprint(rec)
        if fp in hashes: duplicates.append([hashes[fp], rid])
        hashes[fp] = rid
        texts = [d['text'] for d in rec['docs']]
        lengths.append(sum(map(len, texts)))
        docs.update(d['type'] for d in rec['docs'])
        missing += bool(rec.get('dropped_doc_counts'))
        replacement += any('\ufffd' in t for t in texts)
        complete.update(json.dumps(rec.get('input_completeness'), ensure_ascii=False, sort_keys=True) for _ in [0])
        for k in ['적용계약법', '업무구분', '계약방법', '낙찰방법', '소관구분']:
            meta[k][str(rec['meta'].get(k))] += 1
    return {'n': len(lengths), 'text_chars': summary(lengths), 'over_4000': sum(x>4000 for x in lengths), 'over_12000': sum(x>12000 for x in lengths), 'over_24000': sum(x>24000 for x in lengths), 'doc_types': dict(docs), 'meta': dict(meta), 'completeness': dict(complete), 'with_dropped_docs': missing, 'with_replacement_chars': replacement, 'duplicate_ids': duplicate_ids, 'duplicate_doc_content': duplicates}, hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='open')
    parser.add_argument('--output', default='analysis/data_profile.json')
    parser.add_argument('--skip-train', action='store_true')
    args = parser.parse_args()
    root = Path(args.root)
    results, hashes = {}, {}
    for key, path in [('dev', root/'dev.jsonl.gz'), ('sample',root/'data/test.jsonl.gz'), ('unlabeled',root/'train_unlabeled.jsonl.gz')]:
        if key == 'unlabeled' and args.skip_train: continue
        results[key], hashes[key] = profile(path)
    results['cross_split_identical_documents'] = {f'{a}/{b}': [[hashes[a][h], hashes[b][h]] for h in hashes[a].keys() & hashes[b].keys()] for a,b in [('dev','sample'), ('dev','unlabeled')] if a in hashes and b in hashes}
    dev = {r['id']:r for r in records(root/'dev.jsonl.gz')}
    with (root/'dev_labels.csv').open(encoding='utf-8-sig', newline='') as stream: labels = list(csv.DictReader(stream))
    item_table = json.loads((root/'data/항목표.json').read_text(encoding='utf-8'))['항목']
    absence = {10,11,16,18,20}
    results['labels'] = {'rows':len(labels), 'all_zero':sum(all(r[f'v{i}']=='0' for i in range(1,25)) for r in labels), 'items':{}}
    for i in range(1,25):
        positives = [r for r in labels if r[f'v{i}']=='1']
        stats = {'name':item_table[f'v{i}']['항목명'], 'positive_count':len(positives), 'ids':[r['id'] for r in positives], 'evidence_empty':0, 'evidence_nonmatching':[], 'evidence_over500':[], 'evidence_docs':collections.Counter(), 'evidence_retained_prefix':{str(n):0 for n in [4000,12000,24000]}, 'evidence_lengths':[]}
        for row in positives:
            rec = dev[row['id']]
            ev = row.get(f'e{i}', '')
            if not ev: stats['evidence_empty'] += 1; continue
            stats['evidence_lengths'].append(len(ev))
            matched = [d['type'] for d in rec['docs'] if ev in d['text']]
            stats['evidence_docs'].update(matched)
            if not matched: stats['evidence_nonmatching'].append(row['id'])
            if len(ev)>500: stats['evidence_over500'].append(row['id'])
            for n in [4000,12000,24000]:
                # Character-prefix diagnostic, not tokenizer capacity estimate.
                text = '\n'.join(d['text'] for d in rec['docs'])[:n]
                stats['evidence_retained_prefix'][str(n)] += ev in text
        stats['evidence_lengths'] = summary(stats['evidence_lengths'])
        results['labels']['items'][f'v{i}'] = stats
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Wrote {target}')


if __name__ == '__main__': main()
