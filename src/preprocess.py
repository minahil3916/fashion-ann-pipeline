import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split


RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    test_size = params["preprocess"]["test_size"]
    random_seed = params["preprocess"]["random_seed"]

    print("Loading raw data...")

    x_train = np.load(os.path.join(RAW_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(RAW_DIR, "y_train.npy"))
    x_test = np.load(os.path.join(RAW_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(RAW_DIR, "y_test.npy"))

    print("Normalizing pixel values...")

    x_train = x_train.astype("float32") / 255.0 - 0.5
    x_test = x_test.astype("float32") / 255.0 - 0.5

    print("Creating training and validation sets...")

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=test_size,
        random_state=random_seed,
        stratify=y_train
    )

    np.save(os.path.join(PROCESSED_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "x_val.npy"), x_val)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(PROCESSED_DIR, "x_test.npy"), x_test)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    print("Preprocessing completed successfully.")
    print(f"Training set: {x_train.shape}")
    print(f"Validation set: {x_val.shape}")
    print(f"Test set: {x_test.shape}")


if __name__ == "__main__":
    main()