# AI StyleLens

AI StyleLens is a local AI fashion workspace for item detection, visual catalog matching, generated virtual try-on, product Q&A, fashion challenges, chat rooms, and browser-to-browser video consult rooms.

The main runnable app lives in `LENS-TRY_ON` and is served by FastAPI.

## Features

- Upload an image or video and detect fashion items.
- Match detected items against the local 300-image fashion catalog.
- Ask product questions using OpenAI when an API key is configured.
- Generate virtual try-on images when an OpenAI image model is configured.
- Submit fashion challenge looks, like/comment on them, and track local coins.
- Use local chat rooms backed by JSON storage.
- Use WebRTC video consult rooms with FastAPI WebSocket signaling.

## Runtime

The project has been tested with:

```text
Python 3.12.2
```

Use the project virtual environment from the repository root:

```powershell
.\.venv\Scripts\python.exe --version
```

## Run Locally

From the repository root:

```powershell
cd .\LENS-TRY_ON
..\.venv\Scripts\python.exe -m uvicorn server:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

Health check:

```powershell
Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8000/api/health"
```

Expected key values:

```text
yolo_model: best.pt
fashion_detector_loaded: true
catalog_images: 300
```

## OpenAI Setup

OpenAI is optional. Detection, catalog matching, challenges, chat, and video work without an API key.

To enable product Q&A and generated try-on:

```powershell
cd .\LENS-TRY_ON
Copy-Item ".env.example" ".env"
notepad ".env"
```

Set:

```text
OPENAI_API_KEY=your_key_here
OPENAI_IMAGE_MODEL=gpt-image-1
```

Do not commit `.env`. It is ignored by Git.

## Deploy To Render

The repository includes a Render Blueprint:

```text
render.yaml
```

Use these settings if you create the service manually instead of using the Blueprint:

```text
Runtime: Python
Root Directory: LENS-TRY_ON
Build Command: python -m pip install --upgrade pip && python -m pip install -r requirement.txt
Start Command: python -m uvicorn server:app --host 0.0.0.0 --port $PORT
Health Check Path: /api/health
Python Version: 3.12.2
```

Required environment variables on Render:

```text
OPENAI_API_KEY=your_key_here
OPENAI_IMAGE_MODEL=gpt-image-1
YOLO_CONFIG_DIR=/tmp
WARM_ML_ON_STARTUP=false
```

`OPENAI_API_KEY` is optional for basic detection, catalog matching, challenges, chat, and video signaling. It is required for OpenAI Q&A and generated try-on.

Render free instances have limited memory. This app loads PyTorch, Ultralytics YOLO, Timm, and local model weights at startup, so a paid instance may be needed if the free instance runs out of memory during build or boot.

## Model

The app automatically prefers:

```text
LENS-TRY_ON/best.pt
```

The included `best.pt` is a local 10-class fashion detector trained from the existing catalog images.

Classes:

```text
sunglasses, hat, jacket, shirt, pants, shorts, skirt, dress, bag, shoes
```

Training summary from the local CPU run:

```text
epochs: 5
train images: 240
val images: 60
mAP50: 0.42404
mAP50-95: 0.23641
```

This is enough for local end-to-end testing. For stronger real-world detection, train on a larger annotated dataset with people wearing items, varied lighting, backgrounds, poses, and occlusions.

## Dataset And Retraining

The catalog images are stored under:

```text
LENS-TRY_ON/embedding_images/embedding_images/
```

Regenerate the local YOLO dataset:

```powershell
cd .\LENS-TRY_ON
..\.venv\Scripts\python.exe .\prepare_fashion_dataset.py
```

Retrain the detector:

```powershell
cd .\LENS-TRY_ON
..\.venv\Scripts\python.exe .\train_fashion_detector.py --epochs 5 --imgsz 416 --batch 8 --workers 0 --device cpu
```

The training script copies the trained model to:

```text
LENS-TRY_ON/best.pt
```

## Local Data

Runtime files are created under:

```text
LENS-TRY_ON/data/
LENS-TRY_ON/uploads/
```

These folders are ignored by Git and can be recreated by the app.

## Legacy Folders

The old React/Firebase chat/video prototype and old static rewards prototype are no longer required for the local app. Their functionality now lives in the FastAPI app under:

```text
LENS-TRY_ON/static/
```

## Git Safety

Before pushing, verify that local secrets and generated runtime files are not staged:

```powershell
git status --short
```

Do not commit:

```text
.env
.venv/
__pycache__/
LENS-TRY_ON/data/
LENS-TRY_ON/uploads/
LENS-TRY_ON/runs/
LENS-TRY_ON/datasets/generated-fashion-yolo/
```
