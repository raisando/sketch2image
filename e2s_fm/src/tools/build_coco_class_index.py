import json
from pathlib import Path

def build_class_index(instances_json: str, out_json: str):
    ann = json.load(open(instances_json))
    # Map from category_id to category name
    id2name = {cat["id"]: cat["name"] for cat in ann["categories"]}
    
    classes = {}
    for it in ann["annotations"]:
        iid = it["image_id"]
        cname = id2name[it["category_id"]]
        classes.setdefault(iid, set()).add(cname)

    # Convert sets to sorted lists
    for k in classes:
        classes[k] = sorted(classes[k])

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(classes, open(out_json, "w"))
    print("[ok] saved:", out_json)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances_json", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    build_class_index(args.instances_json, args.out_json)
