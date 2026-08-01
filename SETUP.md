# Hướng dẫn thiết lập môi trường và chạy training (cho người mới)

Tài liệu này liệt kê đầy đủ các bước cần thiết để chạy 1 model trong repo này với 1 bộ dữ liệu, dành cho người chưa từng chạy project.

## 0. Cấu trúc thư mục `train/` và bảng ánh xạ model

Toàn bộ script training đã được sắp xếp lại theo vai trò, thay cho hơn chục file `train_*.py` nằm phẳng ở gốc repo trước đây:

```
train/
├── without_external_context/  # Model 1: KHÔNG có nhánh External Context (Ảnh 1)
│   ├── train_pixelcnn_cl.py         # UMT + PixelCNN (image reconstruction) + Contrastive Loss
│   └── train_pixelcnn_cl_mbert.py   # Bản trên, backbone mBERT (BertModel/BertConfig thay vì Roberta)
├── external_context/          # Model 2: CÓ nhánh External Context x̃ (Ảnh 2)
│   └── train_external_context.py    # Model 1 + External Context (retrieval + re-ranking, xem CẢNH BÁO bên dưới)
├── ablations/               # Bỏ bớt 1 thành phần của Model 1 để đo đóng góp từng phần
│   └── train_pixelcnn_wo_cl.py      # Model 1 nhưng bỏ Contrastive Loss
├── baselines/               # Các phương pháp so sánh (không phải đóng góp chính)
│   ├── train_umt.py                     # UMT gốc (không PixelCNN, không CL)
│   ├── train_umt_mbert.py               # UMT gốc, backbone mBERT
│   ├── train_maf.py                     # MAF
│   ├── train_cross_attention_softmax.py
│   ├── train_cross_attention_softmax_gate.py
│   ├── train_cross_attention_crf.py
│   ├── train_cross_attention_crf_gate.py
│   └── train_cross_attention_crf_gate_cl.py
└── legacy/                  # Bản nháp/cũ, KHÔNG dùng cho kết quả chính thức
    ├── train_EXCT_draft.py          # Bản nháp sớm của external-context, đã được thay bằng external_context/train_external_context.py
    └── train_maf_legacy_broken.py   # File cũ, import 1 class không còn tồn tại (MTCCMRobertaForMMTokenClassificationCRF) — hiện KHÔNG chạy được, giữ lại chỉ để tham khảo lịch sử
```

## 1. Yêu cầu môi trường

- Python 3.7+ (khuyến nghị 3.8/3.9, tương thích với `transformers` bản dùng RobertaModel)
- GPU + CUDA (khuyến nghị, vì model dùng ResNet-152 + PhoBERT, chạy CPU sẽ rất chậm)
- Ghi lại chính xác version môi trường thực tế sau khi cài đặt để đảm bảo tái lập (xem mục 6):
  ```bash
  python --version
  pip freeze > environment_freeze.txt
  ```

## 2. Cài đặt dependency

```bash
pip install -r requirements.txt
```Done
transformers
seqeval
boto3
pytorch-crf==0.7.2
pytorch_pretrained_bert==0.4.0
torchvision
```
Ngoài ra cần cài `torch` phù hợp với CUDA của máy (xem https://pytorch.org/get-started/locally/), ví dụ:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 3. Tải pretrained ResNet-152 (bắt buộc cho model multimodal)

```bash
wget https://download.pytorch.org/models/resnet152-b121ed2d.pth -O modules/resnet/resnet152.pth
```
Đường dẫn này tương ứng với tham số `--resnet_root modules/resnet` khi chạy training.

## 4. Chuẩn bị dữ liệu

Model trong `train/without_external_context/train_pixelcnn_cl.py` (kiến trúc `UMT_PixelCNN`) là multimodal — cần **cả text lẫn ảnh**:

- **Text**: 3 file `train.txt`, `dev.txt`, `test.txt` theo format CoNLL (xem `sample_data/VLSP/VLSP2016/` làm ví dụ). Đặt cả 3 file vào cùng 1 thư mục, dùng làm `--data_dir`.
- **Ảnh**: thư mục ảnh tương ứng với ID trong text (`IMGID:...`), dùng làm `--path_image`. Lưu ý: nếu bạn chỉ tải phần text của dataset (không tải ảnh), bạn **chưa thể chạy được bất kỳ script nào trong `train/without_external_context/`, `train/external_context/`, `train/ablations/`, `train/baselines/`** vì tất cả đều là model multimodal, bắt buộc có ảnh. Cần tải thêm `ner_image.zip` tương ứng (ví dụ từ `origin+image/VLSP2016/ner_image.zip` trên HuggingFace) và giải nén vào 1 thư mục riêng.
- Toàn bộ model trong repo này đều multimodal (không có phiên bản text-only). Xem mục 0 để chọn đúng script theo model bạn muốn chạy.

Đặt nhãn (label set) đúng với bộ dữ liệu bạn dùng, ví dụ với VLSP2016:
```bash
export LABELS="B-ORG,B-MISC,I-PER,I-ORG,B-LOC,I-MISC,I-LOC,O,B-PER,X,<s>,</s>"
```
(VLSP2018 và VLSP2021 có label set khác — xem README.md để lấy đúng danh sách.)

## 5. Chạy training

Ví dụ chạy cho VLSP2016 (tham khảo README.md để lấy cấu hình đầy đủ cho từng bộ dữ liệu):

```bash
export LABELS="B-ORG,B-MISC,I-PER,I-ORG,B-LOC,I-MISC,I-LOC,O,B-PER,X,<s>,</s>"

python train/without_external_context/train_pixelcnn_cl.py \
    --do_train \
    --do_eval \
    --output_dir output/vlsp2016_run1 \
    --bert_model "vinai/phobert-base-v2" \
    --alpha 0.5 --beta 0.5 --sigma 0.005 --theta 0.05 \
    --warmup_proportion 0.4 \
    --gradient_accumulation_steps 1 \
    --weight_decay_pixelcnn 0.00005 \
    --lr_pixelcnn 0.001 \
    --learning_rate 2.2e-5 \
    --data_dir "sample_data/VLSP/VLSP2016" \
    --num_train_epochs 10 \
    --train_batch_size 32 \
    --path_image "<đường-dẫn-tới-thư-mục-ảnh>" \
    --task_name "vlsp2016" \
    --resnet_root "modules/resnet" \
    --cache_dir "cache" \
    --max_seq_length 256 \
    --seed 37
```

Kết quả (checkpoint model, checkpoint ResNet encoder, config) sẽ được lưu vào thư mục `--output_dir`.

## 6. Đảm bảo tái lập kết quả (reproducibility)

Các script training đã được cập nhật để cố định seed chặt chẽ hơn (áp dụng cho cả `random`, `numpy`, `torch` CPU/GPU, và tắt tối ưu không xác định của cuDNN):
```python
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
Lưu ý: bật `cudnn.deterministic=True` có thể làm training chậm hơn một chút so với chế độ mặc định, đổi lại kết quả tái lập ổn định hơn giữa các lần chạy trên cùng 1 GPU.

Khi báo cáo kết quả để so sánh model, nên chạy với nhiều seed khác nhau (ví dụ 3–5 seed) và báo cáo mean ± std thay vì 1 lần chạy duy nhất.
