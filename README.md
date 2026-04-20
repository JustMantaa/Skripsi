# SIBI - Sign Language Recognition System

Sistem pengenalan bahasa isyarat real-time menggunakan YOLO untuk deteksi tangan dan LSTM untuk klasifikasi gesture, dengan web interface yang dibangun menggunakan Laravel.

## 📋 Daftar Isi

- [Fitur](#fitur)
- [Struktur Proyek](#struktur-proyek)
- [Prasyarat](#prasyarat)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Dokumentasi API](#dokumentasi-api)
- [Troubleshooting](#troubleshooting)

---

## ✨ Fitur

- **🎥 Real-time Detection**: Deteksi gesture dari webcam atau video file
- **🤖 YOLO + LSTM**: Kombinasi deteksi tangan (YOLO) dan klasifikasi gesture (LSTM)
- **🌐 Web Interface**: Dashboard Laravel untuk monitoring dan management
- **📊 API REST**: API untuk integration dengan aplikasi lain
- **📈 Model Evaluation**: Tools untuk evaluate model performance
- **🎯 Multi-class Support**: Mendukung gesture: J, Z, none, dan extensible

---

## 📂 Struktur Proyek

```
Code/
├── sibi/                          # Laravel Web Application
│   ├── app/                       # Application logic
│   ├── config/                    # Configuration files
│   ├── database/                  # Migrations & seeders
│   ├── resources/                 # Views & assets
│   ├── routes/                    # API & web routes
│   ├── tests/                     # Unit & feature tests
│   ├── composer.json              # PHP dependencies
│   └── package.json               # Node dependencies
│
└── yolo_lstm/                     # Python ML Pipeline
    ├── inference_api.py           # Flask API untuk inference
    ├── realtime_test_yolo.py      # YOLO real-time testing
    ├── realtime_test_lstm.py      # LSTM real-time testing
    ├── hand_landmarker.task       # MediaPipe hand detection model
    ├── requirements_api.txt       # Python dependencies
    │
    ├── training/                  # Training & preprocessing scripts
    │   ├── preprocess.py          # Video normalization
    │   ├── ekstrakKoordinat.py    # Hand landmark extraction
    │   ├── normalisasi_keypoint.py# Coordinate normalization
    │   ├── augment.py             # Data augmentation
    │   ├── split_data.py          # Train/val/test splitting
    │   ├── train_torch.py         # Model training (PyTorch)
    │   └── evaluate_torch.py      # Model evaluation
    │
    └── output/                    # Trained models & datasets
        ├── lstm_model.pt          # LSTM model (PyTorch)
        ├── best.pt                # YOLO model
        ├── lstm_meta_torch.json   # Model metadata
        ├── train.csv              # Training dataset
        ├── val.csv                # Validation dataset
        └── test.csv               # Test dataset
```

---

## 🔧 Prasyarat

### System Requirements
- **OS**: Windows 10+, macOS, Linux
- **Python**: 3.9+
- **Node.js**: 16+ (untuk Laravel frontend)
- **PHP**: 8.1+ (untuk Laravel backend)
- **GPU**: NVIDIA GPU recommended (CUDA 11.8+) untuk performa optimal

### Software
- Git
- Composer (PHP package manager)
- npm atau yarn (Node package manager)

---

## 💾 Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/sibi.git
cd Code
```

### 2. Setup Python Environment (YOLO LSTM)

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r yolo_lstm/requirements_api.txt
```

### 3. Setup Laravel (SIBI)

```bash
cd sibi

# Install PHP dependencies
composer install

# Install Node dependencies
npm install

# Copy environment file
cp .env.example .env

# Generate app key
php artisan key:generate

# Run migrations (jika menggunakan database)
php artisan migrate

# Build assets
npm run build
```

---

## 🚀 Penggunaan

### A. Real-time Testing YOLO

```bash
cd yolo_lstm

# Testing dengan webcam
python realtime_test_yolo.py \
    --model output/best.pt \
    --camera 0 \
    --conf 0.25

# Testing dengan video file
python realtime_test_yolo.py \
    --model output/best.pt \
    --input path/to/video.mp4
```

**Keyboard Controls:**
- `q` - Quit
- `s` - Save frame

### B. Real-time Testing LSTM (Gesture Recognition)

```bash
cd yolo_lstm

# Testing gesture recognition dari webcam
python realtime_test_lstm.py \
    --model output/lstm_model.pt \
    --meta output/lstm_meta_torch.json

# Testing dari video file
python realtime_test_lstm.py \
    --model output/lstm_model.pt \
    --meta output/lstm_meta_torch.json \
    --input path/to/video.mp4
```

### C. Start Flask API Server

```bash
cd yolo_lstm
python inference_api.py
```

API akan berjalan di `http://localhost:5000`

**Example Request:**
```bash
curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"image": "base64_encoded_image"}'
```

### D. Start Laravel Development Server

```bash
cd sibi

# Start development server
php artisan serve

# Di terminal lain, start Vite dev server (untuk hot reload)
npm run dev
```

Laravel akan berjalan di `http://localhost:8000`

---

## 📡 Dokumentasi API

### Flask API Endpoints

#### 1. **POST /predict**
Predict gesture dari image

```bash
Request:
{
    "image": "base64_encoded_image_string"
}

Response:
{
    "gesture": "J",
    "confidence": 0.95,
    "landmarks": [...],
    "timestamp": "2026-04-20T12:34:56"
}
```

#### 2. **GET /health**
Check API health status

```bash
Response:
{
    "status": "ok",
    "version": "1.0.0",
    "models_loaded": {
        "yolo": true,
        "lstm": true
    }
}
```

---

## 🏋️ Training Custom Model

### 1. Prepare Data

Siapkan video raw gesture dalam folder:
```
yolo_lstm/dataset/
├── J/
├── Z/
└── none/
```

### 2. Run Preprocessing Pipeline

```bash
cd yolo_lstm

# Step 1: Normalize videos
python training/preprocess.py \
    --input dataset/ \
    --output temp/normalized

# Step 2: Extract hand landmarks
python training/ekstrakKoordinat.py \
    --input temp/normalized \
    --output temp/landmarks

# Step 3: Normalize coordinates
python training/normalisasi_keypoint.py \
    --input temp/landmarks \
    --output output/dataset_normalized.csv

# Step 4: Data augmentation (optional)
python training/augment.py \
    --input output/dataset_normalized.csv \
    --output output/dataset_augmented.csv

# Step 5: Split data
python training/split_data.py \
    --input output/dataset_augmented.csv \
    --train-ratio 0.7 \
    --val-ratio 0.15
```

### 3. Train Model

```bash
python training/train_torch.py \
    --train-file output/train_aug.csv \
    --val-file output/val.csv \
    --epochs 100 \
    --batch-size 32
```

### 4. Evaluate Model

```bash
python training/evaluate_torch.py \
    --model output/lstm_model.pt \
    --meta output/lstm_meta_torch.json \
    --test-file output/test.csv
```

---

## 📊 Model Architecture

### YOLO (Hand Detection)
- **Input**: Image (RGB, 960x960)
- **Output**: Bounding boxes + confidence scores
- **Framework**: YOLOv8 (Ultralytics)

### LSTM (Gesture Classification)
```
Input: Hand landmarks (21 points × 3 coords) × 30 frames
  ↓
LSTM Layer 1: 64 hidden units
  ↓
Dropout: 0.5
  ↓
LSTM Layer 2: 32 hidden units
  ↓
Dense Layer 1: 32 units → ReLU
  ↓
Dense Layer 2: 16 units → ReLU
  ↓
Output Layer: num_classes (softmax)
```

**Hyperparameters:**
- Learning Rate: 1e-3
- Batch Size: 32
- Epochs: 100
- Optimizer: Adam
- Loss: CrossEntropyLoss

---

## 🐛 Troubleshooting

### Issue: "Model not found" error

```bash
# Ensure model files exist
ls yolo_lstm/output/

# Download pre-trained models jika belum ada
# (Instruksi download di repository)
```

### Issue: Webcam not detected

```bash
# List available cameras
python -c "import cv2; print(cv2.getBuildInformation())"

# Try different camera index
python realtime_test_lstm.py --camera 1
```

### Issue: Out of Memory (OOM)

```bash
# Reduce batch size
python training/train_torch.py --batch-size 16

# Reduce sequence length
# Edit config di train_torch.py: TIMESTEPS = 15
```

### Issue: GPU not detected

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Install correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📚 Dependencies

### Python Packages
```
torch >= 2.0.0
torchvision >= 0.15.0
opencv-python >= 4.8.0
mediapipe >= 0.10.0
numpy >= 1.24.0
pandas >= 1.5.0
scikit-learn >= 1.3.0
flask >= 2.3.0
flask-cors >= 4.0.0
ultralytics >= 8.0.0
```

### PHP Packages (Laravel)
```
laravel/framework: ^10.0
laravel/sanctum: ^3.0
laravel/tinker: ^2.8
```

---

## 📝 Development Notes

### Code Style
- Python: PEP 8 (use `black` formatter)
- PHP: PSR-12 (use `php-cs-fixer`)
- JavaScript: Prettier

### Testing
```bash
# Python tests
pytest yolo_lstm/tests/

# PHP tests
cd sibi && php artisan test
```

---

## 👥 Contributors

- **Developer**: Your Name
- **Advisor**: [Advisor Name]

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📞 Contact & Support

- **Issues**: Create an issue on GitHub
- **Email**: your.email@example.com
- **Documentation**: [Full Documentation Link]

---

## 🔗 Useful Links

- [PyTorch Documentation](https://pytorch.org/docs/)
- [MediaPipe Hand Detection](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [YOLO Documentation](https://docs.ultralytics.com/)
- [Laravel Documentation](https://laravel.com/docs/)

---

**Last Updated**: April 2026
