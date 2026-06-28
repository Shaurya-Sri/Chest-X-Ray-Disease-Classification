# 🫁 Chest X-Ray Pathology Classifier

A deep learning system that classifies chest X-ray images into **20 thoracic pathology categories** using a CNN backbone with transfer learning, served via a FastAPI backend and Streamlit frontend, fully containerized with Docker.


---

## What it detects

| | | | |
|---|---|---|---|
| Atelectasis | Cardiomegaly | Consolidation | Edema |
| Effusion | Emphysema | Fibrosis | Hernia |
| Infiltration | Mass | No Finding | Nodule |
| Pleural Thickening | Pneumonia | Pneumothorax | Enlarged Cardiomediastinum |
| Lung Opacity | Lung Lesion | Fracture | Support Devices |

---

## Project Structure

```
chest-xray-classifier/
├── api/
│   ├── main.py                    # FastAPI inference server
│   └── requirements_api.txt
├── frontend/
│   ├── streamlit_app.py           # Streamlit UI
│   └── requirements_streamlit.txt
├── X-Ray.ipynb                    # Full training pipeline (Kaggle)
├── model.pkl                      # Trained model weights 
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
└── README.md

```

---

## Model

```
Input (224×224 RGB)
        ↓
CNN Backbone (pretrained on ImageNet)
  └── Frozen for first 2 epochs → then fine-tuned
        ↓
Custom Head: Dropout → Linear(512) → ReLU → Linear(20)
        ↓
Sigmoid → 20 class probabilities → Threshold → Predictions
```

**Training details:**

| Setting | Value |
|---|---|
| Loss | Asymmetric BCE (FN penalized 5×) |
| Optimizer | AdamW (lr=1e-4) |
| Scheduler | CosineAnnealingLR |
| Training images | 51,043 |
| Test images | 17,015 |
| Mixed precision | torch.cuda.amp |
| Early stopping | patience=3 |

---

## Scoring Metric

```
Score_c = (TP - FP - 5·FN) / N      per class
Final   = macro-average across all 20 classes
```

Missing a disease (FN) is penalized 5× more than a false alarm (FP), reflecting real clinical cost.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Model status |
| GET | `/classes` | All 20 class names |
| POST | `/predict` | Single image inference |
| POST | `/predict_batch` | Batch inference |


## Run with Docker

```bash
git clone https://github.com/Shaurya-Sri/Chest-X-Ray-Disease-Classification.git
cd Chest-X-Ray-Disease-Classification.git

docker-compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:8501 |
| API | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |

```bash
# Stop
docker-compose down
```

---

## Run Locally

```bash
python -m venv venv
venv\Scripts\activate              


pip install -r api/requirements_api.txt
pip install -r frontend/requirements_streamlit.txt
```

**Terminal 1 — API:**
```bash
cd api
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
streamlit run streamlit_app.py
```

---

## Training (Kaggle)

Open `X-Ray.ipynb` on Kaggle with GPU enabled.

```python
# Install extra deps
!pip install timm albumentations -q

# Update paths in CFG
'data_dir': '/kaggle/input/YOUR-DATASET-NAME',
'img_dir' : '/kaggle/input/YOUR-DATASET-NAME/images',
```

Run all cells — `model.pkl` saves to `/kaggle/working/`. Expected runtime: **~45–60 min** on T4/P100.

---

## Tech Stack

| | |
|---|---|
| Model | PyTorch + timm |
| Augmentation | Albumentations |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker + Docker Compose |
| Training | Kaggle GPU |

---

