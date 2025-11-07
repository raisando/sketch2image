# sketch2image
Tesis Sketch to Image

Para replicar resultados

## 1) git clone

## 1.1) VENV
```bash
python3 -m venv venv \
source venv/bin/activate \
pip install --upgrade pip \
pip install -r requirements.txt --? \
pip install matplotlib \
pip install open_clip_torch \
pip install seaborn \
pip install scikit-learn \
```
## 2) COCO
```bash
~/repo/sketch2image/sketch_cond \ 
mkdir -p data && cd data        
```

### Descarga las imágenes de train y val
```bash
wget http://images.cocodataset.org/zips/train2017.zip \
wget http://images.cocodataset.org/zips/val2017.zip 
```

### Descarga las anotaciones
```bash
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip 
```
### Descomprime
```bash
unzip train2017.zip \
unzip val2017.zip_trainval2017.zip \
unzip annotations_trainval2017.zip 
```
### Limpieza opcional
```bash
rm train2017.zip val2017.zip annotations_trainval2017.zip
```
### Estructura final

 data/ \
 └── coco2017/ \
&nbsp;&nbsp;&nbsp;&nbsp;├── train2017/ \
&nbsp;&nbsp;&nbsp;&nbsp;├── val2017/ \
&nbsp;&nbsp;&nbsp;&nbsp;└── annotations/ 

## Setup
### Build class JSON
#### Get most common class and index on JSON 🔢 by count
🏋️ Train
```bash
python -m src.tools.build_most_common_class_index   --instances_json /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/annotations/instances_train2017.json   --out_json /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/cache/most_common_class_index_train.json
```
🧪 Val
```bash
python -m src.tools.build_most_common_class_index   --instances_json /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/annotations/instances_val2017.json   --out_json /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/cache/most_common_class_index_val.json
```
---

#### Get primary class JSON 🖼️ by bound box
 🏋️ Train
```bash
python -m src.tools.build_most_present_class_index   --instances_json data/coco2017/annotations/instances_train2017.json   --out_json data/coco2017/cache/most_present_class_index_train.json   --min_area_frac 0.33   --use_captions   --captions_json data/coco2017/annotations/captions_train2017.json
```
 🧪 Val
```bash
python -m src.tools.build_most_present_class_index   --instances_json data/coco2017/annotations/instances_val2017.json   --out_json data/coco2017/cache/most_present_class_index_val.json   --min_area_frac 0.33   --use_captions   --captions_json data/coco2017/annotations/captions_val2017.json
```

---
### Build CLIP embeddings
##### 🔢 By count
🏋️ Train
```bash
python -m src.tools.build_clip_embeddings_coco   --captions_index_json data/coco2017/cache/most_common_class_index_train.json   --out_pt data/coco2017/cache/clip_most_common_class_embeds_train.pt
```
 🧪 Val
```bash
python -m src.tools.build_clip_embeddings_coco   --captions_index_json data/coco2017/cache/most_common_class_index_val.json   --out_pt data/coco2017/cache/clip_most_common_class_embeds_val.pt
```

**OR**
##### 🖼️ By bound box
 🏋️ Train
```bash
python -m src.tools.build_clip_embeddings_coco   --captions_index_json data/coco2017/cache/most_present_class_index_train.json   --out_pt data/coco2017/cache/clip_most_present_class_index_embeds_train.pt
```
 🧪 Val
```bash
python -m src.tools.build_clip_embeddings_coco   --captions_index_json data/coco2017/cache/most_present_class_index_val.json   --out_pt data/coco2017/cache/clip_most_present_class_index_embeds_val.pt
```

## 3) Filter to 5 classes
##### 🔢 By count
 🏋️ Train
```bash
python -m  src.tools.filter_coco   --classes "person,car,dog,cat,bicycle"   --json /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/cache/most_common_class_index_train.json   --embeds_pt       /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/cache/clip_most_common_class_embeds_train.pt   --src_images_dir  /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/train2017   --out_root        /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017_5cls   --split train2017 
```
🧪 Val

```bash

python -m  src.tools.filter_coco   --classes "person,car,dog,cat,bicycle"   --json /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/cache/most_common_class_index_val.json   --embeds_pt       /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/cache/clip_most_common_class_embeds_val.pt   --src_images_dir  /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/val2017   --out_root        /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017_5cls   --split val2017 
```
OR
##### 🖼️ By bound box
 🏋️ Train
```bash
python -m  src.tools.filter_coco   --classes "person,car,dog,cat,bicycle"   --json /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/cache/most_present_class_index_train.json   --embeds_pt /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/cache/clip_most_present_class_index_embeds_train.pt   --src_images_dir /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/train2017   --out_root /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017_5cls   --split train2017 
```
🧪 Val
```bash
python -m  src.tools.filter_coco   --classes "person,car,dog,cat,bicycle"   --json /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/cache/most_present_class_index_val.json   --embeds_pt home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/cache/clip_most_present_class_index_embeds_val.pt   --src_images_dir  /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017/val2017   --out_root /home/rsandoval/repo/sketch2image/sketch_cond/data/coco2017_5cls   --split val2017
```

## 4) Train
```bash
torchrun --nproc_per_node=3 -m src.train --data_root data/coco2017_5cls --size 128 --epochs 100 --batch 16 --lr 1e-4 --num_workers 4 --out runs/fm_coco_filtered_s128_b16_e100
```

## 5) Sample 
```bash
python -m src.sample --ckpt runs/fm_coco_filtered_s128_b16_e100/model.pth --size 128 --channels 3 --num 16 --steps 750 --out runs/fm_coco_s128_b16_e100/samples_person.png --prompt person
```

python -m src.sample --ckpt runs/fm_coco_filtered_s128_b16_e100/model.pth --size 128 --channels 3 --num 16 --steps 750 --out runs/fm_coco_s128_b16_e100/samples_bicycle.png --prompt bicycle 

python -m src.sample --ckpt runs/fm_coco_filtered_s128_b16_e100/model.pth --size 128 --channels 3 --num 16 --steps 750 --out runs/fm_coco_s128_b16_e100/samples_dog.png --prompt dog 

## 6) Continued Training
torchrun --nproc_per_node=3 -m src.train_contd --data_root data/coco2017_5cls --size 128 --epochs 100 --batch 16 --lr 1e-4 --num_workers 4 --out runs/fm_coco_filtered_s128_b16_e100 --ckpt_every 10 --val_every 10 --resume runs/fm_coco_filtered_s128_b16_e100/checkpoints/ckpt_e0100.pth
