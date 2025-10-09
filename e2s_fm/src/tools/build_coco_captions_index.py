import json
from pathlib import Path

def build_caption_index(captions_json: str, out_json: str):
    ann = json.load(open(captions_json))
    caps = {}
    for it in ann["annotations"]:
        iid = it["image_id"]
        caps.setdefault(iid, []).append(it["caption"])
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(caps, open(out_json, "w"))
    print("[ok] saved:", out_json)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions_json", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    build_caption_index(args.captions_json, args.out_json)
