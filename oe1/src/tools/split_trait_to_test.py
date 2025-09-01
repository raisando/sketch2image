#!/usr/bin/env python3
"""
Crea un split de test a partir de train, dejando val intacto.
Por defecto mueve 10% de las imágenes de train → test con semilla fija (reproducible).
Soporta .jpg/.png y mantiene formato A|B del pix2pix (no separa mitades).

Uso:
  python tools/split_train_to_test.py --root data/edges2shoes --ratio 0.10 --seed 42
Opciones:
  --copy  (si lo pasas, copia en lugar de mover)
  --exts ".jpg,.png" (si necesitas otro patrón)
"""

from pathlib import Path
import argparse
import random
import shutil
import sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True, help="Raíz del dataset (ej. data/edges2shoes)")
    p.add_argument("--ratio", type=float, default=0.10, help="Fracción de train que irá a test (0.10 = 10%)")
    p.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    p.add_argument("--copy", action="store_true", help="Si se pasa, copia en lugar de mover")
    p.add_argument("--exts", type=str, default=".jpg,.png", help="Extensiones válidas separadas por coma")
    args = p.parse_args()

    root = Path(args.root)
    train_dir = root / "train"
    val_dir   = root / "val"
    test_dir  = root / "test"
    if not train_dir.is_dir():
        print(f"[ERROR] No existe {train_dir}", file=sys.stderr)
        sys.exit(1)
    if not val_dir.is_dir():
        print(f"[WARN] No existe {val_dir}. No es obligatorio, pero se recomienda tener val/ separado.")

    test_dir.mkdir(parents=True, exist_ok=True)

    exts = {e.strip().lower() for e in args.exts.split(",")}
    files = []
    for ext in exts:
        files.extend(sorted(train_dir.glob(f"*{ext}")))
    if not files:
        print(f"[ERROR] No se encontraron imágenes en {train_dir} con extensiones {exts}", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    random.shuffle(files)

    n_total = len(files)
    n_test = max(1, int(n_total * args.ratio))
    test_files = files[:n_test]

    # Manifiesto para reproducibilidad
    manifest = test_dir / "manifest_from_train.txt"
    with manifest.open("w", encoding="utf-8") as f:
        f.write("# Archivos movidos/copied desde train a test\n")
        f.write(f"# root={root}\n# ratio={args.ratio}\n# seed={args.seed}\n")
        for pth in test_files:
            f.write(f"{pth.name}\n")

    op = shutil.copy2 if args.copy else shutil.move
    for src in test_files:
        dst = test_dir / src.name
        if dst.exists():
            print(f"[WARN] Ya existe en test: {dst.name}, saltando…")
            continue
        op(str(src), str(dst))

    n_train_after = len(list(train_dir.glob("*")))
    n_test_after  = len(list(test_dir.glob("*")))
    print(f"[OK] Total train antes: {n_total}")
    print(f"[OK] Movidos a test: {len(test_files)}  (ratio={args.ratio:.2f}, seed={args.seed})")
    print(f"[OK] Ahora train contiene: {n_train_after}  | test contiene: {n_test_after}")
    print(f"[INFO] Manifiesto: {manifest}")

if __name__ == "__main__":
    main()
