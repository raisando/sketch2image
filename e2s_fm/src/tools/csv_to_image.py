import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

def csv_to_images(csv_path, out_dir, img_size=28):
    """
    Convierte CSV de Fashion-MNIST a imágenes PNG organizadas por clase.

    Args:
        csv_path: ruta al CSV (train o test)
        out_dir: carpeta donde se guardarán las imágenes
        img_size: tamaño (28 para MNIST/FashionMNIST)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Leer CSV
    df = pd.read_csv(csv_path)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Procesando {csv_path}"):
        label = int(row.iloc[0])                  # clase (0–9)
        pixels = row.iloc[1:].values.astype(np.uint8)  # pixeles

        # Reshape a (28,28)
        img_array = pixels.reshape(img_size, img_size)

        # Crear carpeta de clase
        label_dir = os.path.join(out_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

        # Guardar como PNG
        img = Image.fromarray(img_array, mode="L")  # "L" = escala de grises
        img.save(os.path.join(label_dir, f"{idx}.png"))

# Uso:
csv_to_images("/Users/raimundosandoval/code/U/sketch2image/sketch2image/e2s_fm/data/fashionmnist/fashion-mnist_train.csv", "images/train")
csv_to_images("/Users/raimundosandoval/code/U/sketch2image/sketch2image/e2s_fm/data/fashionmnist/fashion-mnist_test.csv", "images/test")
