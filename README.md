# X-GRU NIDS Backend

FastAPI backend for the X-GRU network intrusion detection system used in the FYP pipeline. The service accepts CSV traffic data, runs model inference, and exposes explainability and forensic endpoints for the latest upload.

## Features

- CSV upload for traffic analysis
- Intrusion detection inference using the bundled TensorFlow model
- Single-alert explanation endpoint
- Global XAI summary endpoint
- XAI quality summary endpoint
- Forensic summary and case listing endpoints

## Repository Layout

```text
app/
  main.py                FastAPI application and routes
  ml_service.py          Model loading and prediction logic
  analysis_builder.py    Summary and forensic report generation
  analysis_runtime.py    Runtime store for the latest analysis
  xai_service.py         SHAP/LIME explanation engine
  schemas.py             Pydantic response models
artifacts/
  model/                 Trained intrusion detection model
  preprocessing/         Scaler and category mapping artifacts
  sequences/             Sequence configuration metadata
uploads/                 Uploaded CSV files are stored here at runtime
requirements.txt         Python dependencies
```

## Requirements

- Python 3.10 or newer is recommended
- A virtual environment is strongly recommended

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Locally

Start the API with Uvicorn:

```bash
uvicorn app.main:app --reload
```

The server will usually be available at:

```text
http://127.0.0.1:8000
```

Interactive API docs are available at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## API Endpoints

- `GET /` - Health check and service status
- `GET /model-info` - Model metadata and feature information
- `POST /upload-traffic` - Upload a CSV file for intrusion analysis
- `POST /explain-alert` - Generate an explanation for a detected window
- `GET /xai/global-summary` - Get global feature importance summary
- `GET /xai/quality-summary` - Get explanation quality metrics
- `GET /forensic/summary` - Get forensic summary for the latest analysis
- `GET /forensic/cases` - List forensic cases, optionally filtered by label

## Usage Notes

- The backend expects CSV input.
- Uploaded files are saved under `uploads/` for the current session.
- The service keeps the latest upload analysis in memory, so the explanation and forensic endpoints only work after a successful CSV upload.
- The model and preprocessing artifacts in `artifacts/` are required for inference.

## Git Ignore Policy

This repository ignores Python caches, editor settings, build outputs, and uploaded CSV data. The `uploads/` directory is kept with a placeholder file so the backend can create runtime uploads without committing the data itself.

## License

No license has been added yet.
