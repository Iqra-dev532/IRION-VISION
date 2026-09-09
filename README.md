# IRION VISION

**See beyond. Identify the bloom.**

IRION VISION is a Flask web application for real Iris flower image classification. It predicts exactly three species: Iris Setosa, Iris Versicolor, and Iris Virginica. Uploads and browser camera captures share one TensorFlow/Keras model service and every prediction is stored in SQLite by default.

## Architecture

`Browser -> Flask Blueprint -> /api/predict -> app/services/model_service.py -> TensorFlow/Keras -> Prediction SQLAlchemy model -> analytics`

The model is never called directly from a route. `IrisModelService` loads `models/iris_classifier.keras` once and applies the same 224x224 / 0-1 preprocessing path for inference. If the model is absent, the app returns a useful error and never invents a prediction.

## Install and run

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python run.py
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run.py
```

Open `http://127.0.0.1:5000`. Database tables are created automatically during development.

## Dataset and model training

This project requires real Iris flower photographs, not the tabular sklearn Iris dataset. Place licensed/permissioned images in:

```text
dataset/
  train/{setosa,versicolor,virginica}/
  validation/{setosa,versicolor,virginica}/
  test/{setosa,versicolor,virginica}/
```

Then train the MobileNetV2 transfer-learning classifier:

```powershell
python training/train.py
python training/evaluate.py
```

Training creates `models/iris_classifier.keras` and `models/class_names.json`. Evaluation writes the confusion matrix to `training/results/`. A model file is intentionally not included because a real trained model must be produced from a valid photo dataset.

## User flows

- Register and sign in
- Dashboard with database-backed species metrics
- Upload JPG, JPEG, PNG, or WEBP at `/predict/image`
- Capture a frame with browser permissions at `/predict/camera`
- Review and filter history at `/predictions/history`
- View analytics at `/analytics`
- Admin overview at `/admin`

Camera access requires explicit browser permission and normally works on localhost or HTTPS.

## API

Authenticated endpoints:

- `POST /api/predict` with multipart field `image`; optional `source=upload|camera`
- `GET /api/predictions?page=1&species=Iris%20Setosa&source=upload`
- `GET /api/predictions/<id>`
- `DELETE /api/predictions/<id>`
- `GET /api/predictions/<id>/image`
- `GET /api/statistics`
- `GET /api/user`

Public endpoint:

- `GET /api/health`

Example successful response:

```json
{
  "success": true,
  "prediction": "Iris Setosa",
  "confidence": 0.9782,
  "probabilities": {
    "Iris Setosa": 0.9782,
    "Iris Versicolor": 0.0143,
    "Iris Virginica": 0.0075
  },
  "processing_time": 0.24
}
```

## Security and production

Passwords are hashed with Werkzeug. Browser forms use CSRF protection, uploads use extension/MIME checks and secure random filenames, and queries are scoped to the logged-in user. Set a strong `SECRET_KEY`, use PostgreSQL via `DATABASE_URL`, serve with a production WSGI server, keep uploaded files private, and use HTTPS for camera access.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

The test suite verifies registration, login, protected routes, admin protection, user-scoped APIs, and the explicit no-model behavior. Real inference validation requires installing TensorFlow, adding the dataset, and running training first.
