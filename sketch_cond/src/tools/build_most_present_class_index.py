# build_most_present_class_index.py
import json
from pathlib import Path

def build_most_present_class_index(instances_json: str, out_json: str,
                                   min_area_frac: float = 0.05,
                                   whitelist=None,
                                   use_captions: bool = False,
                                   captions_json: str = None):
    """
    Builds a 'most present' class index per image based on the total area
    covered by instances of each category, optionally filtering by area fraction,
    whitelist of classes, and captions mentioning the class.

    Args:
        instances_json (str): Path to COCO instances JSON file.
        out_json (str): Path where output JSON will be saved.
        min_area_frac (float): Minimum fraction of the image area that must
            be covered by the chosen class (default 0.05 = 5%).
        whitelist (list[str] | None): Optional list of allowed class names.
        use_captions (bool): Whether to require that the class name appear
            in at least one of the image captions.
        captions_json (str | None): Path to COCO captions JSON file
            (required if use_captions=True).
    """
    ann = json.load(open(instances_json))
    id2name = {cat["id"]: cat["name"] for cat in ann["categories"]}
    name2id = {v: k for k, v in id2name.items()}

    # Build lookup: image_id -> (width, height)
    imgid_to_size = {im["id"]: (im["width"], im["height"]) for im in ann["images"]}

    # Load captions if requested
    imgid_to_captions = {}
    if use_captions:
        if captions_json is None:
            raise ValueError("use_captions=True requires --captions_json")
        capt = json.load(open(captions_json))
        for c in capt["annotations"]:
            iid = c["image_id"]
            imgid_to_captions.setdefault(iid, []).append(c["caption"].lower())

    # Aggregate total area per (image_id, category_id)
    stats = {}
    for a in ann["annotations"]:
        iid = a["image_id"]
        cid = a["category_id"]
        x, y, w, h = a["bbox"]
        area = w * h
        stats.setdefault(iid, {}).setdefault(cid, 0.0)
        stats[iid][cid] += area

    out = {}
    skipped_tiny, skipped_caption, skipped_whitelist = 0, 0, 0

    for iid, cat_areas in stats.items():
        img_w, img_h = imgid_to_size[iid]
        img_area = img_w * img_h
        if img_area <= 0:
            continue

        best_cid, best_frac = None, 0.0
        for cid, total_area in cat_areas.items():
            frac = total_area / img_area
            if frac > best_frac:
                best_cid, best_frac = cid, frac

        if best_cid is None:
            continue

        best_name = id2name[best_cid]

        # whitelist filter
        if whitelist and best_name not in whitelist:
            skipped_whitelist += 1
            continue

        # minimum area fraction
        if best_frac < min_area_frac:
            skipped_tiny += 1
            continue

        # caption filter
        if use_captions:
            captions = imgid_to_captions.get(iid, [])
            if not any(best_name in c for c in captions):
                skipped_caption += 1
                continue

        out[iid] = best_name

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f)
    print(f"[ok] saved: {out_json}")
    print(f"[summary] total={len(out)}, "
          f"skipped_tiny={skipped_tiny}, "
          f"skipped_caption={skipped_caption}, "
          f"skipped_whitelist={skipped_whitelist}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build most-present class index from COCO annotations.")
    ap.add_argument("--instances_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--min_area_frac", type=float, default=0.05,
                    help="Minimum fraction of image area for main class (default=0.05)")
    ap.add_argument("--whitelist", nargs="+", default=None,
                    help="Optional list of class names to include")
    ap.add_argument("--use_captions", action="store_true",
                    help="Require the class to appear in at least one caption")
    ap.add_argument("--captions_json", type=str, default=None,
                    help="Path to COCO captions JSON (needed if --use_captions)")

    args = ap.parse_args()

    build_most_present_class_index(
        instances_json=args.instances_json,
        out_json=args.out_json,
        min_area_frac=args.min_area_frac,
        whitelist=args.whitelist,
        use_captions=args.use_captions,
        captions_json=args.captions_json,
    )
