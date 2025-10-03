# sketch2image
Tesis Sketch to Image

Para replicar resultados

Test 1)
python -m src.train \
  --data_root /home/raisando/tesis/e2s_fm/data \
  --size 64 --batch 64 --lr 1e-4 \
  --trainer epochs --epochs 200 --patience 999999 \
  --out runs/fm_epochs200

CIFAR 10)
python -m src.train \
  --data_root /home/raisando/tesis/sketch2image/e2s_fm/data/cifar10/images \
  --size 32 \
  --epochs 200 \
  --batch 64 \
  --lr 1e-4 \
  --out runs/fm_cifar10_class0


  python -m src.sample \
  --ckpt runs/fm_cifar10_class0/model.pth \
  --out runs/fm_cifar10_class0/samples.png \
  --size 32 \
  --channels 3 \
  --num 36 \
  --steps 500
