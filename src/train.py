import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf


PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    dense_units = params["train"]["dense_units"]
    dropout_rate = params["train"]["dropout_rate"]
    learning_rate = params["train"]["learning_rate"]
    epochs = params["train"]["epochs"]
    batch_size = params["train"]["batch_size"]

    print("Loading processed data...")

    x_train = np.load(os.path.join(PROCESSED_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    x_val = np.load(os.path.join(PROCESSED_DIR, "x_val.npy"))
    y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))

    print("Building ANN model...")

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(dense_units, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Training model...")

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    model.save(os.path.join(MODEL_DIR, "model.h5"))

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(
        os.path.join(MODEL_DIR, "history.csv"),
        index=False
    )

    print("Training completed successfully.")
    print("Model saved to models/model.h5")
    print("Training history saved to models/history.csv")


if __name__ == "__main__":
    main()