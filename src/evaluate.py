import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
METRICS_FILE = "metrics.json"
CONFUSION_MATRIX_FILE = "confusion_matrix.png"


def main():
    print("Loading test data and trained model...")

    x_test = np.load(os.path.join(PROCESSED_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "model.h5")
    )

    print("Evaluating model...")

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=1
    )

    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot"
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.title("Fashion-MNIST Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        os.path.join(MODEL_DIR, CONFUSION_MATRIX_FILE)
    )
    plt.close()

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy)
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("Confusion matrix saved to models/confusion_matrix.png")
    print("Metrics saved to metrics.json")


if __name__ == "__main__":
    main()