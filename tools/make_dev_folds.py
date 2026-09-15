"""Create deterministic multilabel dev folds without fitting any model.

All split artifacts are local validation material, never private evaluation data.
"""
import argparse, collections, csv, gzip, json, random
from pathlib import Path
from profile_data import records, fingerprint


def assign_folds(rows, k=5, seed=20260910):
    rng=random.Random(seed)
    assignments={}
    remaining=set(range(len(rows)))
    positives={i:{j for j in range(1,25) if row[f'v{j}']=='1'} for i,row in enumerate(rows)}
    needs=[{j:sum(j in v for v in positives.values())/k for j in range(1,25)} for _ in range(k)]
    capacity=[len(rows)/k]*k
    while remaining:
        counts=collections.Counter(j for i in remaining for j in positives[i])
        if not counts: break
        minimum=min(counts.values())
        label=rng.choice(sorted(j for j,v in counts.items() if v==minimum))
        candidates=sorted(i for i in remaining if label in positives[i])
        rng.shuffle(candidates)
        candidates.sort(key=lambda i:-len(positives[i]))
        for i in candidates:
            best_need=max(n[label] for n in needs)
            eligible=[f for f in range(k) if needs[f][label]==best_need]
            best_cap=max(capacity[f] for f in eligible)
            fold=rng.choice([f for f in eligible if capacity[f]==best_cap])
            assignments[i]=fold
            capacity[fold]-=1
            for j in positives[i]: needs[fold][j]-=1
            remaining.remove(i)
    leftovers=sorted(remaining)
    rng.shuffle(leftovers)
    for i in leftovers:
        maximum=max(capacity)
        fold=rng.choice([f for f in range(k) if capacity[f]==maximum])
        assignments[i]=fold
        capacity[fold]-=1
    return assignments


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='open')
    parser.add_argument('--output',default='analysis/dev_folds')
    args=parser.parse_args()
    root, output=Path(args.root),Path(args.output)
    recs=list(records(root/'dev.jsonl.gz'))
    by_id={r['id']:r for r in recs}
    with (root/'dev_labels.csv').open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        columns=reader.fieldnames
        rows=list(reader)
    if {r['id'] for r in rows}!=set(by_id): raise ValueError('Dev input/label IDs differ')
    if len({fingerprint(r) for r in recs})!=len(recs): raise ValueError('Duplicate document content requires grouped splitting')
    for seed in range(20260910,20261910):
        assignment=assign_folds(rows,seed=seed)
        if all(any(assignment[i]==f and r[f'v{j}']=='1' for i,r in enumerate(rows)) for f in range(5) for j in range(1,25)): break
    else: raise ValueError('Could not produce folds with each label represented')
    output.mkdir(parents=True,exist_ok=True)
    manifest={'seed':seed,'method':'iterative multilabel stratification; first seed with positive support for every fold and label; no fitted model','limitation':'Rule design inspected the full dev set. Excluding a retrieval fold is an internal diagnostic, not fully unseen generalization evaluation.','folds':[]}
    for f in range(5):
        selected=[r for i,r in enumerate(rows) if assignment[i]==f]
        support={f'v{j}':sum(r[f'v{j}']=='1' for r in selected) for j in range(1,25)}
        if not all(support.values()): raise ValueError(f'Fold {f} has zero-positive label: {support}')
        ids=[r['id'] for r in selected]
        manifest['folds'].append({'fold':f,'ids':ids,'positive_support':support,'all_zero':sum(all(r[f'v{j}']=='0' for j in range(1,25)) for r in selected),'excluded_retrieval_ids':ids})
        with (output/f'fold_{f}_labels.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns)
            writer.writeheader();writer.writerows(selected)
        with gzip.open(output/f'fold_{f}.jsonl.gz','wt',encoding='utf-8') as stream:
            for rid in ids: stream.write(json.dumps(by_id[rid],ensure_ascii=False)+'\n')
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in f.items() if k not in ('ids','excluded_retrieval_ids')} for f in manifest['folds']],ensure_ascii=False))


if __name__=='__main__': main()
