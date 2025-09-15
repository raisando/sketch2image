#!/usr/bin/env python3
"""
Aumenta (o crea) el split de val tomando imágenes desde train hasta
llegar a un target (por cantidad exacta o por ratio del total train+val+test).
Mantiene el formato A|B del pix2pix (no separa mitades).

Uso típico:
  python tools/rebalance_val_from_train.py --root data/edges2shoes --val_target 2000 --seed 42
Opciones:
  --copy  (si lo pasas, copia en lugar de mover)
  --exts ".jpg,.png" (extensiones válidas)
  --val_ratio 0.05   (alternativa a --val_target: 5% del total)
  --dry_run          (muestra qué haría, no mueve)
"""

from pathlib import Path
import argparse, random, shutil, sys

def list_images(d, exts, recursive=False):
    if not d.exists():
        return []
    it = d.rglob("*") if recursive else d.glob("*")
    return sorted([p for p in it if p.suffix.lower() in exts and p.is_file()])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exts", type=str, default=".jpg,.png")
    p.add_argument("--copy", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--recursive", action="store_true", help="recorrer subcarpetas")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--val_target", type=int, help="cantidad objetivo exacta para val")
    group.add_argument("--val_ratio", type=float, help="proporción objetivo (0-1) del total (train+val+test)")
    args = p.parse_args()

    root = Path(args.root)
    train_dir = root / "train"
    val_dir   = root / "val"
    test_dir  = root / "test"

    for d in [train_dir, val_dir, test_dir]:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)

    exts = {e.strip().lower() for e in args.exts.split(",")}

    train_files = list_images(train_dir, exts, recursive=args.recursive)
    val_files   = list_images(val_dir,   exts, recursive=args.recursive)
    test_files  = list_images(test_dir,  exts, recursive=args.recursive)

    n_train, n_val, n_test = len(train_files), len(val_files), len(test_files)
    n_total = n_train + n_val + n_test

    if args.val_ratio is not None:
        target = int(round(args.val_ratio * n_total))
    else:
        target = args.val_target

    need = max(0, target - n_val)
    print(f"[INFO] total={n_total} | train={n_train} val={n_val} test={n_test}")
    print(f"[INFO] val target={target} | need to move from train → val: {need}")

    if need == 0:
        print("[OK] val ya cumple el objetivo. Nada que hacer.")
        return
    if need > n_train:
        print(f"[ERROR] No hay suficientes imágenes en train para alcanzar val={target}. Falta {need - n_train}.", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    candidates = train_files[:]  # copia
    random.shuffle(candidates)
    pick = candidates[:need]

    op = shutil.copy2 if args.copy else shutil.move
    manifest = val_dir / "manifest_from_train.txt"
    if not args.dry_run:
        with manifest.open("a", encoding="utf-8") as mf:
            mf.write(f"# add {need} from train (seed={args.seed})\n")
            for src in pick:
                dst = val_dir / src.name
                if dst.exists():
                    print(f"[WARN] ya existe en val: {dst.name}, saltando…")
                    continue
                mf.write(f"{src.name}\n")
                op(str(src), str(dst))
        # recuenta
        n_train_new = len(list_images(train_dir, exts, recursive=args.recursive))
        n_val_new   = len(list_images(val_dir,   exts, recursive=args.recursive))
        print(f"[OK] train={n_train_new} | val={n_val_new} | test={n_test}")
        print(f"[INFO] manifiesto: {manifest}")
    else:
        print("[DRY RUN] mover/copiaría los siguientes archivos:")
        for src in pick[:20]:
            print("  -", src.name)
        if need > 20:
            print(f"  … y {need-20} más")
        print("[DRY RUN] no se realizaron cambios.")

if __name__ == "__main__":
    main()
