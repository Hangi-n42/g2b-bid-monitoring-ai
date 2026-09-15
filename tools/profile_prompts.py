"""Token and retained-evidence diagnostics with the exact fixed tokenizer.
No model inference and no classification score is reported.
"""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'submission'))
from script import Assets,read_records,fit_messages,make_messages
from transformers import AutoTokenizer

class Counter:
    def __init__(self):
        self.tok=AutoTokenizer.from_pretrained(ROOT/'analysis/fixed_tokenizer',local_files_only=True)
    def count(self,messages):
        ids=self.tok.apply_chat_template(messages,add_generation_prompt=True,tokenize=True)
        if hasattr(ids,'keys') and 'input_ids' in ids: ids=ids['input_ids']
        return len(ids)

def main():
    runner=Counter()
    assets=Assets(ROOT/'open/data',ROOT/'submission/model')
    recs=read_records(ROOT/'open/dev.jsonl.gz')
    with (ROOT/'open/dev_labels.csv').open(encoding='utf-8-sig',newline='') as f:
        labels={r['id']:r for r in csv.DictReader(f)}
    details=[]
    covered=total=0
    for rec in recs:
        msgs,spans,ids,tokens=fit_messages(rec,assets,runner)
        own=labels[rec['id']]
        for i in range(1,25):
            if own[f'v{i}']=='1' and own[f'e{i}']:
                total+=1
                covered+=any(own[f'e{i}'] in s['text'] for s in spans)
        details.append({'id':rec['id'],'tokens':tokens,'selected_chars':sum(len(s['text']) for s in spans),
                        'examples':len(ids),'retrieved_dev_ids':ids,'spans':len(spans)})
    counts=[r['tokens'] for r in details]
    result={'type':'tokenization and input-retention only; not model execution or F1',
        'fixed_tokenizer_revision':'4d7ae4984b7db7de8f8457170b3f1a419ee76d52',
        'prompt_tokens':{'min':min(counts),'median':statistics.median(counts),'max':max(counts),'sum':sum(counts)},
        'evidence_retained_after_token_fitting':covered,'reference_evidence_count':total,
        'examples_used_counts':{str(i):sum(r['examples']==i for r in details) for i in range(3)},
        'code_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'submission').glob('*.py')},
        'details':details}
    (ROOT/'analysis/prompt_profile.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='details'},ensure_ascii=True,indent=2))
if __name__=='__main__': main()
