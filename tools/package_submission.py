"""Explicit allowlist prevents local data, outputs, secrets or models in ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
FILES=['script.py','context.py','model/item_guides.json','model/examples.json',
       'model/reviewed_examples.json','model/provenance.json']

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default='submit_v4.zip')
    args=parser.parse_args()
    if Path(args.name).name!=args.name or not args.name.endswith('.zip'):
        parser.error('name must be a ZIP filename without directories')
    target=ROOT/'artifacts'
    target.mkdir(exist_ok=True)
    zip_path=target/args.name
    manifest={}
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for rel in FILES:
            p=ROOT/'submission'/rel
            data=p.read_bytes()
            info=zipfile.ZipInfo(rel,date_time=(2026,9,10,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)
            manifest[rel]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
        assert sorted(z.namelist())==sorted(FILES)
        assert sum(x.file_size for x in z.infolist())<8*1024**3
    receipt={'zip':str(zip_path.relative_to(ROOT)),'bytes':zip_path.stat().st_size,
        'sha256':hashlib.sha256(zip_path.read_bytes()).hexdigest(),'files':manifest,
        'actual_fixed_model_execution_verified':False,'submitted':False}
    (target/(zip_path.stem+'_manifest.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=True,indent=2))

if __name__=='__main__': main()
