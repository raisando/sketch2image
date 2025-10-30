import json
from pathlib import Path

def build_most_common_class_index(instances_json: str, out_json: str):
    ann = json.load(open(instances_json))
    id2name = {cat["id"]: cat["name"] for cat in ann["categories"]}

    count_per_image = {}
    for a in ann["annotations"]:
        iid = a["image_id"]
        cid = a["category_id"]
        count_per_image.setdefault(iid, {}).setdefault(cid, 0)
        count_per_image[iid][cid] += 1

    out = {}
    for iid, counts in count_per_image.items():
        # Get category_id with highest count
        majority_cid = max(counts.items(), key=lambda x: x[1])[0]
        out[iid] = id2name[majority_cid]

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_json, "w"))
    print("[ok] saved:", out_json)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances_json", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    build_most_common_class_index(args.instances_json, args.out_json)
