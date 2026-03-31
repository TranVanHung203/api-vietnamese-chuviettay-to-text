# API VietOCR (chữ viết tay -> text)

API này dùng [VietOCR](https://github.com/pbcquoc/vietocr) để nhận diện chữ viết tay/chữ in tiếng Việt từ ảnh.

## 1) Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Nếu bạn dùng GPU CUDA, nên cài `torch`/`torchvision` theo đúng lệnh tại trang chính thức PyTorch trước, rồi mới chạy `pip install -r requirements.txt`.

## 2) Chạy server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Mở docs Swagger: `http://localhost:8000/docs`
Mở trang viết tay test nhanh: `http://localhost:8000/`

## 3) Cấu hình bằng biến môi trường

- `VIETOCR_MODEL`: model name (mặc định `vgg_transformer`)
  - Hỗ trợ: `vgg_transformer`, `vgg_seq2seq`, `resnet_transformer`, `resnet_fpn_transformer`
- `VIETOCR_DEVICE`: ví dụ `cpu` hoặc `cuda:0`
  - Nếu không set: tự chọn `cuda:0` khi có GPU, ngược lại `cpu`
- `VIETOCR_BEAMSEARCH`: `true/false` (mặc định `true`)

Ví dụ chạy CPU:

```bash
set VIETOCR_DEVICE=cpu
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 4) Gọi API

### Health check

```bash
curl http://localhost:8000/health
```

### OCR từ file ảnh

```bash
curl -X POST "http://localhost:8000/ocr/file" \
  -F "file=@test.jpg"
```

Mặc định endpoint bật `multiline=true` để nhận diện ảnh nhiều dòng.
`number_word_mode` mặc định là `false`.
Chỉ khi bật `true` mới chuẩn hóa về nhóm từ đọc số
(`một`, `mười`, `trăm`, `nghìn`, `triệu`, `tỷ`, ...).

### OCR từ base64

```bash
curl -X POST "http://localhost:8000/ocr/base64" \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\":\"<BASE64_IMAGE>\",\"return_prob\":false,\"multiline\":true,\"number_word_mode\":false}"
```

## Lưu ý chất lượng nhận diện

- VietOCR hoạt động tốt nhất với ảnh đã cắt vùng chữ (text line/word), rõ nét, ít nhiễu.
- Nếu ảnh là cả trang giấy, nên tách vùng dòng chữ trước rồi mới đưa vào API để có kết quả tốt hơn.
