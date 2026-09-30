import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist


RAW_DIR = "data/raw"


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("Downloading/loading Fashion-MNIST...")

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    np.save(os.path.join(RAW_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(RAW_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(RAW_DIR, "x_test.npy"), x_test)
    np.save(os.path.join(RAW_DIR, "y_test.npy"), y_test)

    print("Raw data saved successfully.")
    print(f"Training images: {x_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Testing images: {x_test.shape}")
    print(f"Testing labels: {y_test.shape}")


if __name__ == "__main__":
    main()