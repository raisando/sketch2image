# sketch2image
Tesis Sketch to Image

Para replicar resultados

## 1) git clone

## 1.1) VENV
python3 -m venv venv \
source venv/bin/activate \
pip install --upgrade pip \
pip install -r requirements.txt --? \
pip install matplotlib \
pip install open_clip_torch \
pip install seaborn \
pip install scikit-learn \

## 2) COCO 
~/repo/sketch2image/sketch_cond \ 
mkdir -p data && cd data        

### Descarga las imágenes de train y val
wget http://images.cocodataset.org/zips/train2017.zip \
wget http://images.cocodataset.org/zips/val2017.zip 

### Descarga las anotaciones
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip 

### Descomprime
unzip train2017.zip \
unzip val2017.zip_trainval2017.zip \
unzip annotations_trainval2017.zip 

### Limpieza opcional
rm train2017.zip val2017.zip annotations_trainval2017.zip

### Estructura final

 data/ \
 └── coco2017/ \
&nbsp;&nbsp;&nbsp;&nbsp;├── train2017/ \
&nbsp;&nbsp;&nbsp;&nbsp;├── val2017/ \
&nbsp;&nbsp;&nbsp;&nbsp;└── annotations/ 

## Setup
### Get most commond class and index on json
python -m src.tools.build_most_common_class_index --instances_json /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/annotations/instances_train2017.json --out_json /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/cache/most_common_class_index_train.json \

python -m src.tools.build_most_common_class_index --instances_json /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/annotations/instances_val2017.json --out_json /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/cache/most_common_class_index_val.json 

### Build CLIP embeddings
python -m src.tools.build_clip_embeddings_coco --captions_index_json data/coco2017/cache/most_common_class_index_train.json   --out_pt data/coco2017/cache/clip_most_common_class_embeds_train.pt \
python -m src.tools.build_clip_embeddings_coco --captions_index_json data/coco2017/cache/most_common_class_index_val.json   --out_pt data/coco2017/cache/clip_most_common_class_embeds_val.pt 

## 3) Filter to 5 classes
python -m  src.tools.filter_coco   --classes "person,car,dog,cat,bicycle"   --most_common_json /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/cache/most_common_class_index_train.json   --embeds_pt       /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/cache/clip_most_common_class_embeds_train.pt   --src_images_dir  /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/train2017   --out_root        /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017_5cls   --split train2017 

python -m  src.tools.filter_coco   --classes "person,car,dog,cat,bicycle"   --most_common_json /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/cache/most_common_class_index_val.json   --embeds_pt       /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/cache/clip_most_common_class_embeds_val.pt   --src_images_dir  /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/val2017   --out_root        /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017_5cls   --split val2017 

## 4) Train
torchrun --nproc_per_node=3 -m src.train --data_root data/coco2017_5cls --size 128 --epochs 100 --batch 16 --lr 1e-4 --num_workers 4 --out runs/fm_coco_filtered_s128_b16_e100

## 5) Sample 
python -m src.sample --ckpt runs/fm_coco_filtered_s128_b16_e100/model.pth --size 128 --channels 3 --num 16 --steps 750 --out runs/fm_coco_s128_b16_e100/samples_person.png --prompt person 

python -m src.sample --ckpt runs/fm_coco_filtered_s128_b16_e100/model.pth --size 128 --channels 3 --num 16 --steps 750 --out runs/fm_coco_s128_b16_e100/samples_bicycle.png --prompt bicycle 

python -m src.sample --ckpt runs/fm_coco_filtered_s128_b16_e100/model.pth --size 128 --channels 3 --num 16 --steps 750 --out runs/fm_coco_s128_b16_e100/samples_dog.png --prompt dog 

## 6) Continued Training
torchrun --nproc_per_node=3 -m src.train_contd --data_root data/coco2017_5cls --size 128 --epochs 100 --batch 16 --lr 1e-4 --num_workers 4 --out runs/fm_coco_filtered_s128_b16_e100 --ckpt_every 10 --val_every 10 --resume runs/fm_coco_filtered_s128_b16_e100/checkpoints/ckpt_e0100.pth
