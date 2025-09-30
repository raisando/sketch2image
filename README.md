# sketch2image
Tesis Sketch to Image

Para replicar resultados

Test 1)
python -m src.train \
  --data_root /home/raisando/tesis/e2s_fm/data \
  --size 64 --batch 64 --lr 1e-4 \
  --trainer epochs --epochs 200 --patience 999999 \
  --out runs/fm_epochs200
