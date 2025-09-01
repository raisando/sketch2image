from pathlib import Path
from PIL import Image
import numpy as np

def sketch_stats(img_path, to_gray=True):
    im = Image.open(img_path)
    if to_gray:
        im = im.convert("L")  # 1 canal
    arr = np.array(im)  # 0..255
    uniq = np.unique(arr)
    return {
        "path": str(img_path),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "n_unique": len(uniq),
        "unique_first_10": uniq[:10].tolist()
    }

root = Path("/home/raisando/tesis/oe1/data/edges2shoes/test")  # o train/val
files = sorted(list(root.glob("*.jpg")) + list(root.glob("*.png")))[:50]

for f in files[:5]:
    print(sketch_stats(f))

from PIL import Image

def left_half_sketch_stats(img_path):
    im = Image.open(img_path).convert("L")
    w, h = im.size
    sketch = im.crop((0, 0, w//2, h))  # izquierda
    arr = np.array(sketch)
    uniq = np.unique(arr)
    return {"path": str(img_path), "min": int(arr.min()), "max": int(arr.max()),
            "n_unique": len(uniq), "unique_first_10": uniq[:10].tolist()}

print(left_half_sketch_stats(files[0]))


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

im = Image.open(files[0]).convert("L")
w,h = im.size
sketch = im.crop((0,0,w//2,h))
arr = np.array(sketch).ravel()  # 0..255

plt.figure()
plt.hist(arr, bins=256)
plt.title("Histograma de intensidades (sketch)")
plt.show()
