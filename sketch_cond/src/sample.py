# sample.py
import argparse, torch
from pathlib import Path
from src.model.unet import FMUNetCOCO  # <- usa el mismo que entrenaste
from src.tools.utils import sample_and_save
import torch.nn as nn
import open_clip


# --- CLIP ---
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass",
    "cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot",
    "hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

class NegatedModel(nn.Module):
    def __init__(self, inner): super().__init__(); self.inner = inner
    def forward(self, x, t, y=None): return -self.inner(x, t, y)

def build_clip_text_emb(device, prompt: str, num) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    with torch.no_grad():
        tok = open_clip.tokenize([prompt]).to(device)
        emb  = model.encode_text(tok)             # [N, D]
        emb  = emb / emb.norm(dim=-1, keepdim=True)

    if emb.size(0) == 1 and num > 1:
        y = emb.repeat(num, 1)                 # un prompt → repetir
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="ruta al checkpoint .pth")
    ap.add_argument("--out", type=str, default="samples.png")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--num", type=int, default=36)
    ap.add_argument("--steps", type=int, default=750)

    # --- Condición ---
    ap.add_argument("--prompt", type=str, default=None, help='Texto CLIP, ej. "a photo of a cat"')
    ap.add_argument("--coco_class", type=str, default=None,
                    help="Nombre o id (0-79) de clase COCO; ej. 'dog' o '17'")

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- construye la condición (y) ---
    cond_y = None
    if args.prompt is not None or args.coco_class is not None:
        if args.coco_class is not None and args.prompt is None:
            # permitir id o nombre
            try:
                idx = int(args.coco_class)
                name = COCO_CLASSES[idx]
            except ValueError:
                name = args.coco_class
            prompt = f"{name}"
        else:
            prompt = args.prompt
        cond_y = build_clip_text_emb(device, prompt, args.num)  # [1,512]
        print(f"[cond] prompt = {prompt}")

    # --- definir el UNet EXACTO del training ---
    unet = FMUNetCOCO(
        channels=[32, 64, 128, 256, 512],   # <- pon aquí lo que usaste al entrenar
        num_residual_layers=2,
        t_embed_dim=40, y_embed_dim=40,
        in_channels=args.channels, out_channels=args.channels,
        clip_dim=512,                       # <- importante si condicionaste con CLIP
        num_classes=1
    ).to(device)

    # opcional: si tu sampler necesita el campo negado
    # model = NegatedModel(unet)
    model = unet

    # --- cargar pesos ---
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"[ok] cargado {args.ckpt}")

    # --- samplear (cond_y se replica adentro si B>1) ---
    sample_and_save(
        model, Path(args.out),
        num=args.num, size=args.size, channels=args.channels,
        steps=args.steps, device=device,
        cond_y=cond_y
    )
    print(f"[ok] guardado {args.out}")

if __name__ == "__main__":
    main()
