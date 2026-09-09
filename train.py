import json
from pathlib import Path

import tensorflow as tf

from preprocessing import CLASS_NAMES, IMAGE_SIZE, make_datasets

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def build_model():
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
        tf.keras.layers.RandomContrast(0.1),
    ], name="iris_augmentation")
    backbone = tf.keras.applications.MobileNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    train, validation, _ = make_datasets()
    model = build_model()
    train_counts = [
        len(list((Path("dataset") / "train" / class_name).glob("*")))
        for class_name in ("setosa", "versicolor", "virginica")
    ]
    total = sum(train_counts)
    class_weight = {
        index: total / (len(train_counts) * count)
        for index, count in enumerate(train_counts)
        if count
    }
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3),
        tf.keras.callbacks.ModelCheckpoint(MODEL_DIR / "iris_classifier.keras", monitor="val_accuracy", save_best_only=True),
    ]
    model.fit(train, validation_data=validation, epochs=30, callbacks=callbacks, class_weight=class_weight)
    (MODEL_DIR / "class_names.json").write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
