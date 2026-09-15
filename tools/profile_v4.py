"""Profile real group prompts with the fixed tokenizer, never run a model."""
import argparse
from collections import Counter as Counts
import csv
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'submission'))
import script
from transformers import AutoTokenizer


class TokenCounter:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(ROOT / 'analysis/fixed_tokenizer', local_files_only=True)

    def count(self, messages):
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if hasattr(ids, 'keys') and 'input_ids' in ids:
            ids = ids['input_ids']
        return len(ids)


def hashes():
    paths = list((ROOT / 'submission').glob('*.py')) + list((ROOT / 'submission/model').glob('*.json'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--examples', type=int, default=2)
    parser.add_argument('--max-chars', type=int, default=7000)
    parser.add_argument('--output', default='analysis/v4_prompt_profile.json')
    args = parser.parse_args()
    before = hashes()
    runner = TokenCounter()
    assets = script.Assets(ROOT / 'open/data', ROOT / 'submission/model')
    recs = script.read_records(ROOT / 'open/dev.jsonl.gz')
    with (ROOT / 'open/dev_labels.csv').open(encoding='utf-8-sig', newline='') as stream:
        labels = {r['id']: r for r in csv.DictReader(stream)}
    review_path = ROOT / 'submission/model/reviewed_examples.json'
    reviewed = json.loads(review_path.read_text(encoding='utf-8')) if review_path.exists() else []
    reviewed_by_id = {}
    for example in reviewed:
        reviewed_by_id.setdefault(example['id'], []).append(example)
    calls, missing, review_retention, leakages = [], [], [], []
    usage, negative_usage, review_usage = Counts(), Counts(), Counts()
    expected = retained = 0
    budget_arg = 'prompt_budget' if 'prompt_budget' in inspect.signature(script.fit_messages).parameters else 'budget'
    for rec in recs:
        for name, items, budget in script.GROUPS:
            messages, spans, ids, tokens = script.fit_messages(
                rec, assets, runner, args.max_chars, args.examples, items=items, **{budget_arg: budget})
            assert tokens <= budget
            context = '\n'.join(f'[S{s["sid"]} {s["type"]}] {s["text"]}' for s in spans)
            shown = assets.examples(rec, context, len(ids), items=items)
            assert ids == [e['id'] for e in shown]
            for example in shown:
                if example['id'] == rec['id'] or example['fingerprint'] == script.fingerprint(rec):
                    leakages.append({'id': rec['id'], 'group': name, 'example_id': example['id']})
                for key, label in example['partial_labels'].items():
                    usage[key] += 1
                    if label == 0:
                        negative_usage[key] += 1
                if example.get('example_id'):
                    review_usage[example['example_id']] += 1
            own = labels[rec['id']]
            for item in items:
                evidence = own[f'e{item}']
                if own[f'v{item}'] == '1' and evidence:
                    expected += 1
                    found = any(evidence in s['text'] for s in spans)
                    retained += found
                    if not found:
                        missing.append({'id': rec['id'], 'group': name, 'item': f'v{item}'})
            for example in reviewed_by_id.get(rec['id'], []):
                for key, anchor in example['evidence'].items():
                    item = int(key.lstrip('ve'))
                    if item in items:
                        review_retention.append({'id': rec['id'], 'example_id': example.get('example_id'),
                            'group': name, 'item': f'v{item}', 'label': example['partial_labels'][f'v{item}'],
                            'anchor_retained': any(anchor in s['text'] for s in spans)})
            calls.append({'id': rec['id'], 'group': name, 'tokens': tokens,
                'selected_chars': sum(len(s['text']) for s in spans), 'spans': len(spans),
                'retrieved_ids': ids, 'labels_shown': [e['partial_labels'] for e in shown]})
    totals = [sum(c['tokens'] for c in calls if c['id'] == r['id']) for r in recs]
    after = hashes()
    report = {'type': 'fixed-tokenizer input/retrieval diagnostic only; no model, no F1',
        'examples_requested': args.examples, 'source_max_chars': args.max_chars,
        'groups': script.GROUPS, 'records': len(recs), 'requests': len(calls),
        'input_tokens': {'sum': sum(totals), 'mean_per_notice': statistics.mean(totals),
            'min_per_notice': min(totals), 'max_per_notice': max(totals),
            'by_group': {g: sum(c['tokens'] for c in calls if c['group'] == g) for g, _, _ in script.GROUPS}},
        'reference_evidence': {'retained': retained, 'total': expected, 'missing': missing},
        'reviewed_anchor_retention': review_retention,
        'reviewed_demonstration_usage': dict(review_usage),
        'shown_label_counts': dict(usage), 'shown_negative_label_counts': dict(negative_usage),
        'self_leakages': leakages, 'code_assets_unchanged_during_profile': before == after,
        'hashes': before, 'calls': calls}
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('calls', 'hashes')}, ensure_ascii=True, indent=2))
    if before != after:
        raise RuntimeError('Source/assets changed during profiling; rerun after edits settle')
    if leakages:
        raise RuntimeError('Self answer retrieval detected')


if __name__ == '__main__':
    main()
