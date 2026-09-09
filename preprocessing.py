import shutil
from pathlib import Path

import tensorflow as tf

CLASS_NAMES = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
RAW_DATASET_DIR = Path("iris Computer vision Dataset")


def prepare_dataset(dataset_dir="dataset", raw_dir=RAW_DATASET_DIR):
    root = Path(dataset_dir)
    raw_root = Path(raw_dir)
    source_classes = {
        "setosa": "iris-setosa",
        "versicolor": "iris-versicolour",
        "virginica": "iris-virginica",
    }
    if not raw_root.exists():
        return
    if all((root / split / class_name).is_dir() for split in ("train", "validation", "test") for class_name in source_classes):
        return
    for split in ("train", "validation", "test"):
        for class_name in source_classes:
            (root / split / class_name).mkdir(parents=True, exist_ok=True)
    for class_name, source_name in source_classes.items():
        images = sorted((raw_root / source_name).glob("*.jpg"))
        if len(images) < 3:
            raise FileNotFoundError(f"Not enough images found in {raw_root / source_name}")
        train_end = max(1, int(len(images) * 0.70))
        validation_end = max(train_end + 1, int(len(images) * 0.85))
        for split, split_images in (
            ("train", images[:train_end]),
            ("validation", images[train_end:validation_end]),
            ("test", images[validation_end:]),
        ):
            for image in split_images:
                destination = root / split / class_name / image.name
                if not destination.exists():
                    shutil.copy2(image, destination)


def make_datasets(dataset_dir="dataset"):
    root = Path(dataset_dir)
    prepare_dataset(dataset_dir)
    required_splits = ("train", "validation", "test")
    required_classes = ("setosa", "versicolor", "virginica")
    missing = [
        root / split / class_name
        for split in required_splits
        for class_name in required_classes
        if not (root / split / class_name).is_dir()
    ]
    if missing:
        expected = "dataset/{train,validation,test}/{setosa,versicolor,virginica}/"
        raise FileNotFoundError(
            f"Iris photo dataset is incomplete. Add real images under {expected}; "
            f"missing: {missing[0]}"
        )
    train = tf.keras.utils.image_dataset_from_directory(root / "train", labels="inferred", label_mode="categorical", class_names=["setosa", "versicolor", "virginica"], image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True, seed=42)
    validation = tf.keras.utils.image_dataset_from_directory(root / "validation", labels="inferred", label_mode="categorical", class_names=["setosa", "versicolor", "virginica"], image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False)
    test = tf.keras.utils.image_dataset_from_directory(root / "test", labels="inferred", label_mode="categorical", class_names=["setosa", "versicolor", "virginica"], image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False)
    return train.prefetch(tf.data.AUTOTUNE), validation.prefetch(tf.data.AUTOTUNE), test.prefetch(tf.data.AUTOTUNE)
