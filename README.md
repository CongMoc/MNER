# MNER Vietnamese

## Guide
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Training](#training)

## Requirements
The project is based on PyTorch 1.1+ and Python 3.7. To run our code, install:

```
pip install -r requirements.txt
```

Download the pre-trained ResNet-152 via this link (https://download.pytorch.org/models/resnet152-b121ed2d.pth)
```
wget https://download.pytorch.org/models/resnet152-b121ed2d.pth -O modules/resnet/resnet152.pth
```
## Datasets
vlsp [here](https://vlsp.org.vn/vlsp2021/eval/ner)

### Data Format
The dataset is structured in a specific format, and you can find sample data in the sample_data folder for reference. Ensure your model or processing pipeline is compatible with this format.

## Training

Run:


export LABELS="B-ORG,B-MISC,I-PER,I-ORG,B-LOC,I-MISC,I-LOC,O,B-PER,X,<s>,</s>"
task_name="sonba"
alpha=0.5
beta=0.5
theta=0.05
sigma=0.005
lr_pixelcnn=0.001
weight_decay_pixelcnn=0.00005
learning_rate=2.2e-5
num_train_epochs=10
train_batch_size=32
path_image="/home/vms/bags/vlsp_all/origin+image/VLSP2016/ner_image"
bert_model="vinai/phobert-base-v2"
data_dir="/home/vms/bags/vlsp_all/origin+image/VLSP2016"
resnet_root="modules/resnet"
cache_dir="cache"
max_seq_length=256

python train/main_models/train_pixelcnn_cl.py \
    --do_train \
    --do_eval \
    --output_dir train_umt_pixelcnn_fixedlr_2016_beta${beta}_theta${theta}_sigma${sigma}_lr${learning_rate}_fixcrf \
    --bert_model "${bert_model}" \
    --alpha ${alpha} \
    --beta ${beta} \
    --sigma ${sigma} \
    --theta ${theta} \
    --warmup_proportion 0.4 \
    --gradient_accumulation_steps 1 \
    --weight_decay_pixelcnn ${weight_decay_pixelcnn} \
    --lr_pixelcnn ${lr_pixelcnn} \
    --learning_rate ${learning_rate} \
    --data_dir "${data_dir}" \
    --num_train_epochs ${num_train_epochs} \
    --train_batch_size ${train_batch_size} \
    --path_image "${path_image}" \
    --task_name "${task_name}" \
    --resnet_root "${resnet_root}" \
    --cache_dir "${cache_dir}" \
    --max_seq_length ${max_seq_length}





export LABELS="I-ORGANIZATION,B-ORGANIZATION,I-LOCATION,B-MISCELLANEOUS,I-PERSON,O,B-PERSON,I-MISCELLANEOUS,B-LOCATION,X,<s>,</s>"
task_name="sonba"
alpha=0.5
beta=0.5
theta=0.05
sigma=0.005
lr_pixelcnn=0.001
weight_decay_pixelcnn=0.00005
learning_rate=2.2e-5
num_train_epochs=10
train_batch_size=32
path_image="/home/vms/bags/vlsp_all/origin+image/VLSP2018/ner_image"
bert_model="vinai/phobert-base-v2"
data_dir="/home/vms/bags/vlsp_all/origin+image/VLSP2018"
resnet_root="modules/resnet"
cache_dir="cache"
max_seq_length=256

python train/main_models/train_pixelcnn_cl.py \
    --do_train \
    --do_eval \
    --output_dir train_umt_pixelcnn_fixedlr_2018_beta${beta}_theta${theta}_sigma${sigma}_lr${learning_rate} \
    --bert_model "${bert_model}" \
    --alpha ${alpha} \
    --beta ${beta} \
    --sigma ${sigma} \
    --theta ${theta} \
    --warmup_proportion 0.4 \
    --gradient_accumulation_steps 1 \
    --weight_decay_pixelcnn ${weight_decay_pixelcnn} \
    --lr_pixelcnn ${lr_pixelcnn} \
    --learning_rate ${learning_rate} \
    --data_dir "${data_dir}" \
    --num_train_epochs ${num_train_epochs} \
    --train_batch_size ${train_batch_size} \
    --path_image "${path_image}" \
    --task_name "${task_name}" \
    --resnet_root "${resnet_root}" \
    --cache_dir "${cache_dir}" \
    --max_seq_length ${max_seq_length}





export LABELS="I-PRODUCT-AWARD,B-MISCELLANEOUS,B-QUANTITY-NUM,B-ORGANIZATION-SPORTS,B-DATETIME,I-ADDRESS,I-PERSON,I-EVENT-SPORT,B-ADDRESS,B-EVENT-NATURAL,I-LOCATION-GPE,B-EVENT-GAMESHOW,B-DATETIME-TIMERANGE,I-QUANTITY-NUM,I-QUANTITY-AGE,B-EVENT-CUL,I-QUANTITY-TEM,I-PRODUCT-LEGAL,I-LOCATION-STRUC,I-ORGANIZATION,B-PHONENUMBER,B-IP,O,B-QUANTITY-AGE,I-DATETIME-TIME,I-DATETIME,B-ORGANIZATION-MED,B-DATETIME-SET,I-EVENT-CUL,B-QUANTITY-DIM,I-QUANTITY-DIM,B-EVENT,B-DATETIME-DATERANGE,I-EVENT-GAMESHOW,B-PRODUCT-AWARD,B-LOCATION-STRUC,B-LOCATION,B-PRODUCT,I-MISCELLANEOUS,B-SKILL,I-QUANTITY-ORD,I-ORGANIZATION-STOCK,I-LOCATION-GEO,B-PERSON,B-PRODUCT-COM,B-PRODUCT-LEGAL,I-LOCATION,B-QUANTITY-TEM,I-PRODUCT,B-QUANTITY-CUR,I-QUANTITY-CUR,B-LOCATION-GPE,I-PHONENUMBER,I-ORGANIZATION-MED,I-EVENT-NATURAL,I-EMAIL,B-ORGANIZATION,B-URL,I-DATETIME-TIMERANGE,I-QUANTITY,I-IP,B-EVENT-SPORT,B-PERSONTYPE,B-QUANTITY-PER,I-QUANTITY-PER,I-PRODUCT-COM,I-DATETIME-DURATION,B-LOCATION-GPE-GEO,B-QUANTITY-ORD,I-EVENT,B-DATETIME-TIME,B-QUANTITY,I-DATETIME-SET,I-LOCATION-GPE-GEO,B-ORGANIZATION-STOCK,I-ORGANIZATION-SPORTS,I-SKILL,I-URL,B-DATETIME-DURATION,I-DATETIME-DATE,I-PERSONTYPE,B-DATETIME-DATE,I-DATETIME-DATERANGE,B-LOCATION-GEO,B-EMAIL,X,<s>,</s>"
task_name="sonba"
alpha=0.5
beta=0.5
theta=0.05
sigma=0.005
lr_pixelcnn=0.001
weight_decay_pixelcnn=0.00005
learning_rate=3e-5
num_train_epochs=12
train_batch_size=16
path_image="/home/vms/bags/vlsp_all/origin+image/VLSP2021/ner_image"
bert_model="vinai/phobert-base-v2"
data_dir="/home/vms/bags/vlsp_all/origin+image/VLSP2021"
resnet_root="modules/resnet"
cache_dir="cache"
max_seq_length=256

python train/baselines/train_umt.py \
    --do_train \
    --do_eval \
    --output_dir train_umt_2021_beta${beta}_theta${theta}_sigma${sigma}_lr${learning_rate} \
    --bert_model "${bert_model}" \
    --warmup_proportion 0.4 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${learning_rate} \
    --data_dir "${data_dir}" \
    --num_train_epochs ${num_train_epochs} \
    --train_batch_size ${train_batch_size} \
    --path_image "${path_image}" \
    --task_name "${task_name}" \
    --resnet_root "${resnet_root}" \
    --cache_dir "${cache_dir}" \
    --max_seq_length ${max_seq_length}






































































































```bash

export LABELS="B-ORG,B-MISC,I-PER,I-ORG,B-LOC,I-MISC,I-LOC,O,B-PER,X,<s>,</s>"
task_name="sonba"
alpha=0.5
beta=0.5
theta=0.05
sigma=0.005
lr_pixelcnn=0.001
weight_decay_pixelcnn=0.00005
learning_rate=3e-5
num_train_epochs=10
train_batch_size=64
path_image="/home/rad/nlp/bags/vlsp_all/origin+image/VLSP2016/ner_image"
bert_model="vinai/phobert-base-v2"
data_dir="/home/rad/nlp/bags/vlsp_all/origin+image/VLSP2016"
resnet_root="modules/resnet"
cache_dir="cache"
max_seq_length=256

python train/main_models/train_pixelcnn_cl.py \
    --do_train \
    --do_eval \
    --output_dir $_beta${beta}_theta${theta}_sigma${sigma}_lr${learning_rate} \
    --bert_model "${bert_model}" \
    --alpha ${alpha} \
    --beta ${beta} \
    --sigma ${sigma} \
    --theta ${theta} \
    --warmup_proportion 0.4 \
    --gradient_accumulation_steps 8 \
    --weight_decay_pixelcnn ${weight_decay_pixelcnn} \
    --lr_pixelcnn ${lr_pixelcnn} \
    --learning_rate ${learning_rate} \
    --data_dir "${data_dir}" \
    --num_train_epochs ${num_train_epochs} \
    --train_batch_size ${train_batch_size} \
    --path_image "${path_image}" \
    --task_name "${task_name}" \
    --resnet_root "${resnet_root}" \
    --cache_dir "${cache_dir}" \
    --max_seq_length ${max_seq_length}
```




export LABELS="I-ORGANIZATION,B-ORGANIZATION,I-LOCATION,B-MISCELLANEOUS,I-PERSON,O,B-PERSON,I-MISCELLANEOUS,B-LOCATION,X,<s>,</s>"
task_name="sonba"
alpha=0.5
beta=0.5
theta=0.05
sigma=0.005
lr_pixelcnn=0.001
weight_decay_pixelcnn=0.00005
learning_rate=3e-5
num_train_epochs=10
train_batch_size=32
path_image="/home/rad/nlp/bags/vlsp_all/origin+image/VLSP2018/ner_image"
bert_model="vinai/phobert-base-v2"
data_dir="/home/rad/nlp/bags/vlsp_all/origin+image/VLSP2018"
resnet_root="modules/resnet"
cache_dir="cache"
max_seq_length=256

python train/main_models/train_pixelcnn_cl.py \
    --do_train \
    --do_eval \
    --output_dir $_beta${beta}_theta${theta}_sigma${sigma}_lr${learning_rate} \
    --bert_model "${bert_model}" \
    --alpha ${alpha} \
    --beta ${beta} \
    --sigma ${sigma} \
    --theta ${theta} \
    --warmup_proportion 0.4 \
    --gradient_accumulation_steps 8 \
    --weight_decay_pixelcnn ${weight_decay_pixelcnn} \
    --lr_pixelcnn ${lr_pixelcnn} \
    --learning_rate ${learning_rate} \
    --data_dir "${data_dir}" \
    --num_train_epochs ${num_train_epochs} \
    --train_batch_size ${train_batch_size} \
    --path_image "${path_image}" \
    --task_name "${task_name}" \
    --resnet_root "${resnet_root}" \
    --cache_dir "${cache_dir}" \
    --max_seq_length ${max_seq_length}



export LABELS="I-PRODUCT-AWARD,B-MISCELLANEOUS,B-QUANTITY-NUM,B-ORGANIZATION-SPORTS,B-DATETIME,I-ADDRESS,I-PERSON,I-EVENT-SPORT,B-ADDRESS,B-EVENT-NATURAL,I-LOCATION-GPE,B-EVENT-GAMESHOW,B-DATETIME-TIMERANGE,I-QUANTITY-NUM,I-QUANTITY-AGE,B-EVENT-CUL,I-QUANTITY-TEM,I-PRODUCT-LEGAL,I-LOCATION-STRUC,I-ORGANIZATION,B-PHONENUMBER,B-IP,O,B-QUANTITY-AGE,I-DATETIME-TIME,I-DATETIME,B-ORGANIZATION-MED,B-DATETIME-SET,I-EVENT-CUL,B-QUANTITY-DIM,I-QUANTITY-DIM,B-EVENT,B-DATETIME-DATERANGE,I-EVENT-GAMESHOW,B-PRODUCT-AWARD,B-LOCATION-STRUC,B-LOCATION,B-PRODUCT,I-MISCELLANEOUS,B-SKILL,I-QUANTITY-ORD,I-ORGANIZATION-STOCK,I-LOCATION-GEO,B-PERSON,B-PRODUCT-COM,B-PRODUCT-LEGAL,I-LOCATION,B-QUANTITY-TEM,I-PRODUCT,B-QUANTITY-CUR,I-QUANTITY-CUR,B-LOCATION-GPE,I-PHONENUMBER,I-ORGANIZATION-MED,I-EVENT-NATURAL,I-EMAIL,B-ORGANIZATION,B-URL,I-DATETIME-TIMERANGE,I-QUANTITY,I-IP,B-EVENT-SPORT,B-PERSONTYPE,B-QUANTITY-PER,I-QUANTITY-PER,I-PRODUCT-COM,I-DATETIME-DURATION,B-LOCATION-GPE-GEO,B-QUANTITY-ORD,I-EVENT,B-DATETIME-TIME,B-QUANTITY,I-DATETIME-SET,I-LOCATION-GPE-GEO,B-ORGANIZATION-STOCK,I-ORGANIZATION-SPORTS,I-SKILL,I-URL,B-DATETIME-DURATION,I-DATETIME-DATE,I-PERSONTYPE,B-DATETIME-DATE,I-DATETIME-DATERANGE,B-LOCATION-GEO,B-EMAIL,X,<s>,</s>"
task_name="sonba"
alpha=0.5
beta=0.5
theta=0.05
sigma=0.005
lr_pixelcnn=0.001
weight_decay_pixelcnn=0.00005
learning_rate=3e-5
num_train_epochs=10
train_batch_size=32
path_image="/home/rad/nlp/bags/vlsp_pth/VLSP2021/ner_image"
bert_model="vinai/phobert-base-v2"
data_dir="/home/rad/nlp/bags/vlsp_pth/VLSP2021"
resnet_root="modules/resnet"
cache_dir="cache"
max_seq_length=256

python train/ablations/train_pixelcnn_wo_cl.py \
    --do_train \
    --do_eval \
    --output_dir $_beta${beta}_theta${theta}_sigma${sigma}_lr${learning_rate} \
    --bert_model "${bert_model}" \
    --alpha ${alpha} \
    --beta ${beta} \
    --sigma ${sigma} \
    --theta ${theta} \
    --warmup_proportion 0.4 \
    --gradient_accumulation_steps 8 \
    --weight_decay_pixelcnn ${weight_decay_pixelcnn} \
    --lr_pixelcnn ${lr_pixelcnn} \
    --learning_rate ${learning_rate} \
    --data_dir "${data_dir}" \
    --num_train_epochs ${num_train_epochs} \
    --train_batch_size ${train_batch_size} \
    --path_image "${path_image}" \
    --task_name "${task_name}" \
    --resnet_root "${resnet_root}" \
    --cache_dir "${cache_dir}" \
    --max_seq_length ${max_seq_length}