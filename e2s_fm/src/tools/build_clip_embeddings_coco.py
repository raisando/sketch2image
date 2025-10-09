# src/tools/build_clip_text_embeds.py
import json, torch
from pathlib import Path
import open_clip

def build_clip_text_embeds(captions_index_json: str, out_pt: str,
                           model_name="ViT-B-32", pretrained="openai", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    caps = json.load(open(captions_index_json))
    out = {}
    with torch.no_grad():
        for iid_str, texts in caps.items():
            toks = tokenizer(texts).to(device)
            emb  = model.encode_text(toks)             # [N, D]
            emb  = emb / emb.norm(dim=-1, keepdim=True)
            out[int(iid_str)] = emb.mean(dim=0).float().cpu()
    Path(out_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_pt)
    print("[ok] saved:", out_pt)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions_index_json", required=True)
    ap.add_argument("--out_pt", required=True)
    ap.add_argument("--model_name", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    args = ap.parse_args()
    build_clip_text_embeds(args.captions_index_json, args.out_pt,
                           args.model_name, args.pretrained)
