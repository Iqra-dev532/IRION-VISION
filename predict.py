import sys

from app.services.model_service import IrisModelService


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python training/predict.py path/to/image.jpg")
    service = IrisModelService("models/iris_classifier.keras", "models/class_names.json")
    print(service.predict_image(sys.argv[1]))
