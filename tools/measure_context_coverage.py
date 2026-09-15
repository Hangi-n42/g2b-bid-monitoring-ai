"""Measure exact provided-dev evidence coverage, not prediction accuracy."""
import argparse, csv, importlib.util, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from submission.context import segment_record, select_segments
from profile_data import records


def coverage(rec, evidence, spans):
    best={'one_span_full':False,'union_full':False,'any_overlap':False,'max_single_overlap_ratio':0.0,'union_overlap_ratio':0.0}
    for doc,d in enumerate(rec['docs']):
        start=d['text'].find(evidence)
        while start>=0:
            end=start+len(evidence)
            covered=set()
            for s in spans:
                if s['doc']!=doc: continue
                a,b=max(start,s['start']),min(end,s['end'])
                if b>a:
                    covered.update(range(a,b))
                    best['any_overlap']=True
                    best['max_single_overlap_ratio']=max(best['max_single_overlap_ratio'],(b-a)/len(evidence))
                    best['one_span_full'] |= a==start and b==end
            best['union_overlap_ratio']=max(best['union_overlap_ratio'],len(covered)/len(evidence))
            best['union_full'] |= len(covered)==len(evidence)
            start=d['text'].find(evidence,start+1)
    return best


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='open')
    parser.add_argument('--output',default='analysis/context_coverage.json')
    parser.add_argument('--budget',type=int,default=14000)
    args=parser.parse_args()
    root=Path(args.root)
    spec=importlib.util.spec_from_file_location('provided_baseline',root/'baseline/script.py')
    baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    recs={r['id']:r for r in records(root/'dev.jsonl.gz')}
    with (root/'dev_labels.csv').open(encoding='utf-8-sig',newline='') as stream: labels=list(csv.DictReader(stream))
    results=[]
    for row in labels:
        rec=recs[row['id']]
        candidates=segment_record(rec)
        selected=select_segments(rec,max_chars=args.budget)
        initial=baseline.build_context(rec,max_chars=4000)
        for i in range(1,25):
            ev=row[f'e{i}']
            if row[f'v{i}']!='1' or not ev: continue
            results.append({'id':row['id'],'item':f'v{i}','evidence_length':len(ev),'baseline4000_full':ev in initial,'all_spans':coverage(rec,ev,candidates),'selected_spans':coverage(rec,ev,selected)})
    report={'description':'Input evidence retention only; 68 nonabsence positive labels lack reference evidence and cannot enter this diagnostic. No classification score.','budget':args.budget,'evidence_count':len(results),'baseline4000_full':sum(r['baseline4000_full'] for r in results),'all_spans':{},'selected_spans':{},'details':results}
    for key in ['all_spans','selected_spans']:
        report[key]={m:sum(r[key][m] for r in results) for m in ['one_span_full','union_full','any_overlap']}
        report[key]['mean_max_single_overlap_ratio']=sum(r[key]['max_single_overlap_ratio'] for r in results)/len(results)
        report[key]['mean_union_overlap_ratio']=sum(r[key]['union_overlap_ratio'] for r in results)/len(results)
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='details'},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
