# Phân loại ảnh động vật bằng CNN với TensorFlow/Keras

Dự án phân loại 10 loài trong [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10), so sánh **CNN tự xây dựng** với **CNN ResNet50 học chuyển giao**. Toàn bộ mã chính nằm trong [animals10.ipynb](animals10.ipynb): EDA, chia dữ liệu, pipeline `tf.data`, huấn luyện Keras, đánh giá, Grad-CAM và demo Gradio.

## Cài đặt trên Windows (CPU)

Dùng Python 3.11–3.13 64-bit. Môi trường hiện tại đã kiểm tra với Python 3.13. Trong PowerShell tại thư mục dự án:

```powershell
# Bỏ qua dòng này nếu đã có .venv.
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Trong VS Code, mở notebook, chọn **Select Kernel → Python Environments → .venv**, rồi **Restart Kernel** sau khi cài thư viện. Chạy các ô từ trên xuống. TensorFlow từ 2.11 không hỗ trợ CUDA trực tiếp trên Windows; danh sách GPU rỗng là bình thường khi chạy bằng Python Windows. Xem [hướng dẫn TensorFlow chính thức](https://www.tensorflow.org/install/pip).

## Dùng RTX 4050 qua WSL2

Để huấn luyện với GPU NVIDIA, cài WSL2/Ubuntu và driver NVIDIA hỗ trợ WSL theo [hướng dẫn Microsoft](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl). Mở dự án bằng VS Code **WSL: Reopen Folder in WSL**, rồi chạy trong terminal Ubuntu:

```bash
# Tạo môi trường Linux riêng; không dùng lại .venv của Windows.
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt "tensorflow[and-cuda]>=2.20,<3"
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Chọn kernel `.venv-wsl` trong notebook. Lệnh kiểm tra phải liệt kê GPU trước khi bắt đầu huấn luyện. Nếu không có GPU, kiểm tra `nvidia-smi` trong WSL và [hướng dẫn cài TensorFlow cho WSL2](https://www.tensorflow.org/install/pip#windows-wsl2). Notebook bật memory growth để TensorFlow cấp VRAM theo nhu cầu. Batch size mặc định là 8; giảm xuống 4 nếu hết bộ nhớ.

## Dữ liệu và chạy notebook

Ở ô cấu hình:

- Giữ `DOWNLOAD_DATASET = True` để tải qua kagglehub. Lần đầu cần Internet; nếu dịch vụ yêu cầu xác thực, cấu hình Kaggle theo hướng dẫn của kagglehub.
- Hoặc đặt `DOWNLOAD_DATASET = False` và sửa `DATA_DIR` trỏ trực tiếp tới thư mục `raw-img` chứa `cane`, `cavallo`, `elefante`, `farfalla`, `gallina`, `gatto`, `mucca`, `pecora`, `ragno`, `scoiattolo`.
- Điều chỉnh `CUSTOM_EPOCHS`, `HEAD_EPOCHS`, `FINETUNE_EPOCHS`, `BATCH_SIZE` trước khi chạy. ResNet50 tải trọng số ImageNet lần đầu; huấn luyện đầy đủ trên CPU có thể mất rất lâu.
- Kết quả lưu trong `runs/tensorflow/`. Đổi `RUNS` để giữ riêng từng thí nghiệm; chạy lại cùng đường dẫn sẽ ghi đè kết quả.

Notebook kiểm tra ảnh lỗi và chia phân tầng **70% train / 15% validation / 15% test** với cùng seed. Hai mô hình dùng chung `data/splits.csv`. Pillow đọc RGB và sửa EXIF, resize 224×224; `tf.data` tạo batch dạng NHWC. Chỉ train có lật ngang, xoay nhẹ, thay đổi độ sáng/tương phản. CNN cơ sở chuẩn hóa về [0, 1]; ResNet50 dùng [preprocess_input của Keras](https://keras.io/api/applications/resnet/). Giữ đúng preprocessing khi dự đoán.

CNN cơ sở gồm bốn khối Conv2D → BatchNormalization → ReLU → MaxPooling2D, sau đó GlobalAveragePooling2D, Dropout và Dense 10 logits. ResNet50 huấn luyện classifier trước, sau đó mở nhóm `conv5_*` để fine-tune với learning rate thấp hơn; BatchNormalization luôn đóng băng. Cả hai dùng AdamW và sparse cross-entropy từ logits.

## Kết quả và demo

- `data/`: `splits.csv`, `eda.png`, `invalid_images.txt`.
- `custom/` và `resnet50/`: `best.keras`, `metadata.json`, `history.json`, `learning_curves.png`. Checkpoint có validation loss thấp nhất trong toàn bộ các giai đoạn. Nếu bật `USE_CLASS_WEIGHTS`, cả train và validation dùng trọng số tính từ train, còn accuracy không trọng số.
- `custom/test/` và `resnet50/test/`: `metrics.json`, `confusion_matrix.png`, `errors.csv`; thêm `top_errors.png` và `gradcam.png` nếu có dự đoán sai. Grad-CAM dùng `tf.GradientTape` theo logit dự đoán.
- Ô Gradio cuối notebook: chọn `DEMO_MODEL = "custom"` hoặc `"resnet50"` sau khi huấn luyện. Demo nạp `best.keras` cùng `metadata.json`, hiển thị top 3 nhãn tiếng Việt. Gọi `demo.close()` để dừng server.

Checkpoint `.keras` cần đi kèm `metadata.json` lưu kiến trúc, thứ tự lớp và kích thước ảnh. Trọng số `.pt` và kết quả trong `runs/notebook/` thuộc phiên bản cũ; cần huấn luyện lại bằng TensorFlow. Notebook đã xóa đầu ra cũ để tránh nhầm với kết quả TensorFlow. Các lệnh `animals.py`/`app.py` của hướng dẫn cũ không còn áp dụng; thao tác trực tiếp trong notebook.

## Kiểm tra mã sau khi cài thư viện

```powershell
.\.venv\Scripts\python.exe test_notebook.py
```

Bài kiểm tra dùng ảnh tổng hợp trong thư mục tạm, không tải Animals-10 hay trọng số ImageNet. Kiểm tra pipeline, huấn luyện ngắn cả hai CNN, checkpoint qua hai giai đoạn, nạp lại/dự đoán, đánh giá và Grad-CAM. Kết quả này chỉ xác nhận mã hoạt động, không đo chất lượng mô hình trên dữ liệu thật.

Chỉ dùng validation để chọn mô hình và siêu tham số; test dành cho báo cáo cuối cùng. Chia theo ảnh chưa ngăn ảnh trùng/gần trùng lọt qua các tập. Resize có thể làm méo ảnh; mô hình không có lớp “ngoài 10 loài”, và xác suất cao không bảo đảm đúng. Độ chính xác phải được đo sau khi huấn luyện thực tế.
