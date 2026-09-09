from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from preprocessing import CLASS_NAMES, make_datasets


def main():
    _, _, test = make_datasets()
    model = tf.keras.models.load_model("models/iris_classifier.keras")
    probabilities = model.predict(test)
    predicted = np.argmax(probabilities, axis=1)
    actual = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test])
    print(classification_report(actual, predicted, target_names=CLASS_NAMES, digits=4))
    results = Path("training/results")
    results.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay(confusion_matrix(actual, predicted), display_labels=CLASS_NAMES).plot(cmap="Blues", xticks_rotation=25)
    plt.tight_layout()
    plt.savefig(results / "confusion_matrix.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
