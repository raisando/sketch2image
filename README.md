# sketch2image
Tesis Sketch to Image

Para replicar resultados

## 1) git clone

## 1.1) VENV
python3 -m venv venv \
source venv/bin/activate \
pip install --upgrade pip \
pip install -r requirements.txt \

## 2) COCO 
~/repo/sketch2image \ 
mkdir -p data && cd data \       

### Descarga las imágenes de train y val
wget http://images.cocodataset.org/zips/train2017.zip \
wget http://images.cocodataset.org/zips/val2017.zip \

### Descarga las anotaciones
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip \

### Descomprime
unzip train2017.zip \
unzip val2017.zip_trainval2017.zip \
unzip annotations_trainval2017.zip \

### Limpieza opcional
rm train2017.zip val2017.zip annotations_trainval2017.zip

### Estructura final

 data/ \
 └── coco2017/ \
     ├── train2017/ \
     ├── val2017/ \
     └── annotations/ \

## 3) Filter to 5 classes
 python -m  src.tools.filter_coco   --classes "person,car,dog,cat,bicycle"   --most_common_json /home/rsandoval/repo/sketch2image/e2s_fm/data/coco2017/cache/most_common_class_index_train.json   --embeds_pt       /home/rsandoval/repo/sketch2image/e2s_fm/data/coco2017/cache/clip_most_common_class_embeds_train.pt   --src_images_dir  /home/rsandoval/repo/sketch2image/e2s_fm/data/coco2017/images/train2017   --out_root        /home/rsandoval/repo/sketch2image/e2s_fm/data/coco2017_5cls   --split train2017
  
