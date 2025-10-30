#!/usr/bin/env python3
import json, argparse, shutil, torch
from pathlib import Path

def coco_img_name(img_id: int) -> str:
    # Formato COCO: 12 dígitos con ceros a la izquierda
    return f"{int(img_id):012d}.jpg"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", required=True,
                    help="Lista separada por comas, p.ej.: person,car,dog,cat,bicycle")
    ap.add_argument("--most_common_json", required=True,
                    help="JSON {image_id(str/int): class_name(str)}")
    ap.add_argument("--embeds_pt", required=True,
                    help=".pt con dict {image_id(int): tensor[D]}")
    ap.add_argument("--src_images_dir", required=True,
                    help="Raíz de imágenes COCO: .../train2017")
    ap.add_argument("--out_root", required=True,
                    help="Directorio de salida (se creará).../coco2017_5cls")
    ap.add_argument("--split", default="train2017")
    ap.add_argument("--copy", action="store_true",
                    help="Si no se pasa, hace symlinks en vez de copiar.")
    args = ap.parse_args()

    keep_classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    print("[info] keep_classes =", keep_classes)

    most_common = json.load(open(args.most_common_json))
    # Normalizar llaves a int
    most_common = {int(k): v for k, v in most_common.items()}

    # 1) IDs a conservar
    keep_ids = [iid for iid, cname in most_common.items() if cname in keep_classes]
    keep_ids_set = set(keep_ids)
    print(f"[info] total imgs en json: {len(most_common)} | a conservar: {len(keep_ids)}")

    # 2) Filtrar embeds .pt
    embeds = torch.load(args.embeds_pt, map_location="cpu")
    # algunos .pt pueden tener llaves str o int; normalizamos:
    embeds_norm = {}
    for k, v in embeds.items():
        try:
            kid = int(k)
        except:
            continue
        if kid in keep_ids_set:
            embeds_norm[kid] = v
    print(f"[info] embeds originales: {len(embeds)} | filtrados: {len(embeds_norm)}")

    # 3) Salidas
    out_root = Path(args.out_root)
    out_imgs_dir = out_root / args.split
    out_cache_dir = out_root / "cache"
    out_root.mkdir(parents=True, exist_ok=True)
    out_imgs_dir.mkdir(parents=True, exist_ok=True)
    out_cache_dir.mkdir(parents=True, exist_ok=True)

    # 4) Escribir caches filtrados
    filt_json = {iid: most_common[iid] for iid in keep_ids}
    json_path = out_cache_dir / f"most_common_class_index_{args.split}_5cls.json"
    with json_path.open("w") as f:
        json.dump(filt_json, f, indent=2)
    print("[ok] saved:", json_path)

    embeds_path = out_cache_dir / f"clip_most_common_class_embeds_{args.split}_5cls.pt"
    torch.save(embeds_norm, embeds_path)
    print("[ok] saved:", embeds_path)

    # 5) Symlink/copiar imágenes
    src_images_dir = Path(args.src_images_dir)
    missing, done = 0, 0
    for iid in keep_ids:
        name = coco_img_name(iid)
        src = src_images_dir / name
        dst = out_imgs_dir / name
        if not src.exists():
            missing += 1
            continue
        if args.copy:
            shutil.copy2(src, dst)
        else:
            # symlink relativo (limpia si ya existe)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
        done += 1

    print(f"[ok] imágenes vinculadas/copied: {done} | faltantes: {missing}")
    # 6) Guardar lista de ids por conveniencia
    with (out_cache_dir / f"image_ids_{args.split}_5cls.json").open("w") as f:
        json.dump(sorted(list(keep_ids_set)), f)
    print("[ok] saved:", out_cache_dir / f"image_ids_{args.split}_5cls.json")

if __name__ == "__main__":
    main()
