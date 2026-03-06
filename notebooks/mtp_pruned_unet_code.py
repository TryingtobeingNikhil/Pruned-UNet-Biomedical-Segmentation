# Extracted code from mtp-2-feb.ipynb (pruned U-Net, training, pruning, quantization)

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model, clone_model


# -----------------------------
# Metrics
# -----------------------------

def iou(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - intersection
    return tf.reduce_mean((intersection + smooth) / (union + smooth))


def dice(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    return tf.reduce_mean(
        (2.0 * intersection + smooth)
        / (
            tf.reduce_sum(y_true, axis=[1, 2, 3])
            + tf.reduce_sum(y_pred, axis=[1, 2, 3])
            + smooth
        )
    )


def precision(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    predicted_positives = tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return tf.reduce_mean(true_positives / (predicted_positives + smooth))


def recall(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    actual_positives = tf.reduce_sum(y_true, axis=[1, 2, 3])
    return tf.reduce_mean(true_positives / (actual_positives + smooth))


CUSTOM_OBJECTS = {
    "iou": iou,
    "dice": dice,
    "precision": precision,
    "recall": recall,
}


# -----------------------------
# Light U-Net definition
# -----------------------------

def light_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x1 = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x1 = layers.Conv2D(32, 3, padding="same", activation="relu")(x1)
    p1 = layers.MaxPooling2D((2, 2))(x1)

    x2 = layers.Conv2D(64, 3, padding="same", activation="relu")(p1)
    x2 = layers.Conv2D(64, 3, padding="same", activation="relu")(x2)
    p2 = layers.MaxPooling2D((2, 2))(x2)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(p2)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)

    # Decoder
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, x2])
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, x1])
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    return model


# -----------------------------
# Data loading utilities
# -----------------------------

def process_path(image_path, mask_path, img_size=(256, 256)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, img_size, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(
        mask, img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.where(mask > 0.5, 1.0, 0.0)
    return image, mask


def create_dataset(img_files, msk_files, batch_size=8, img_size=(256, 256)):
    dataset = tf.data.Dataset.from_tensor_slices((img_files, msk_files))
    dataset = dataset.map(
        lambda img, msk: process_path(img, msk, img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)


def load_unified_dataset(
    image_dir, mask_dir, batch_size=8, img_size=(256, 256), train_ratio=0.8, val_ratio=0.1
):
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if len(image_files) != len(mask_files):
        raise ValueError("The number of images and masks does not match.")

    num_files = len(image_files)
    train_split = int(train_ratio * num_files)
    val_split = int((train_ratio + val_ratio) * num_files)

    train_ds = create_dataset(
        image_files[:train_split],
        mask_files[:train_split],
        batch_size=batch_size,
        img_size=img_size,
    )
    val_ds = create_dataset(
        image_files[train_split:val_split],
        mask_files[train_split:val_split],
        batch_size=batch_size,
        img_size=img_size,
    )
    test_ds = create_dataset(
        image_files[val_split:],
        mask_files[val_split:],
        batch_size=batch_size,
        img_size=img_size,
    )
    return train_ds, val_ds, test_ds


# -----------------------------
# Training, pruning, quantization utilities
# -----------------------------

def compile_unet(model, lr=1e-3):
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", iou, dice, precision, recall],
    )
    return model


class CustomPruningCallback(tf.keras.callbacks.Callback):
    def __init__(self, pruning_percent=0.5):
        super().__init__()
        self.pruning_percent = pruning_percent

    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            if hasattr(layer, "kernel"):
                weights = layer.kernel.numpy()
                threshold = np.percentile(
                    np.abs(weights), self.pruning_percent * 100
                )
                pruned_weights = np.where(np.abs(weights) < threshold, 0, weights)
                layer.kernel.assign(pruned_weights)


def prune_model(model, pruning_percent=0.5):
    pruned_model = clone_model(model)
    pruned_model.set_weights(model.get_weights())
    for layer in pruned_model.layers:
        if hasattr(layer, "kernel"):
            w = layer.kernel.numpy()
            threshold = np.percentile(np.abs(w), pruning_percent * 100)
            pruned_w = np.where(np.abs(w) < threshold, 0, w)
            layer.kernel.assign(pruned_w)
    return pruned_model


def simulate_quantization(model, bit_width, quantization_type="int"):
    if bit_width == 32:
        cloned = clone_model(model)
        cloned.set_weights(model.get_weights())
        return cloned

    cloned_model = clone_model(model)
    cloned_model.set_weights(model.get_weights())

    for layer in cloned_model.layers:
        for attr in ["kernel", "bias"]:
            if hasattr(layer, attr):
                weight_var = getattr(layer, attr)
                w = weight_var.numpy()
                if np.all(w == 0):
                    continue
                q_levels = 2**bit_width
                if quantization_type == "int":
                    max_abs = np.max(np.abs(w))
                    if max_abs == 0:
                        continue
                    scale = max_abs / ((q_levels / 2) - 1)
                    quantized = np.clip(
                        np.round(w / scale), -q_levels / 2, q_levels / 2 - 1
                    )
                    dequantized = quantized * scale
                else:
                    w_min, w_max = w.min(), w.max()
                    if w_max - w_min == 0:
                        continue
                    scale = (w_max - w_min) / (q_levels - 1)
                    quantized = np.round((w - w_min) / scale)
                    dequantized = quantized * scale + w_min
                weight_var.assign(dequantized)

    return cloned_model


def convert_to_tflite_int8(model, representative_data):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def convert_to_tflite_float16(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def custom_quantize_model(model, bit_width):
    cloned_model = clone_model(model)
    cloned_model.set_weights(model.get_weights())
    for layer in cloned_model.layers:
        for attr in ["kernel", "bias"]:
            if hasattr(layer, attr):
                w = getattr(layer, attr).numpy()
                if np.all(w == 0):
                    continue
                w_min, w_max = w.min(), w.max()
                q_levels = 2**bit_width
                if w_max == w_min:
                    continue
                scale = (w_max - w_min) / (q_levels - 1)
                q = np.round((w - w_min) / scale)
                w_quant = q * scale + w_min
                getattr(layer, attr).assign(w_quant)
    return cloned_model


def convert_custom_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def evaluate_tflite_model(tflite_model, test_ds):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    expected_dtype = input_details[0]["dtype"]
    scale, zero_point = input_details[0].get("quantization", (1.0, 0))

    all_preds = []
    all_gts = []

    for images, masks in test_ds:
        images_np = images.numpy()
        masks_np = masks.numpy()
        if expected_dtype == np.int8:
            images_np = np.round(images_np / scale) + zero_point
            images_np = images_np.astype(np.int8)

        batch_size = images_np.shape[0]
        for i in range(batch_size):
            sample = np.expand_dims(images_np[i], axis=0)
            interpreter.set_tensor(input_details[0]["index"], sample)
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]["index"])
            all_preds.append(pred[0])
            all_gts.append(masks_np[i])

    all_preds = np.concatenate(
        [p if p.ndim == 4 else np.expand_dims(p, 0) for p in all_preds], axis=0
    )
    all_gts = np.concatenate(
        [g if g.ndim == 4 else np.expand_dims(g, 0) for g in all_gts], axis=0
    )

    binary_preds = (all_preds > 0.5).astype(np.float32)
    accuracy = np.mean(binary_preds == all_gts)
    intersection = np.sum(all_gts * binary_preds)
    union = np.sum(all_gts) + np.sum(binary_preds) - intersection
    iou_val = (intersection + 1e-6) / (union + 1e-6)
    dice_val = (2 * intersection + 1e-6) / (
        np.sum(all_gts) + np.sum(binary_preds) + 1e-6
    )
    return accuracy, iou_val, dice_val


# Note: The original notebook hard-coded Windows paths like:
#   D:\Downloads\MTP-2\unified_dataset\images
# For reuse in this repo, adapt `image_dir` and `mask_dir` to your dataset path.

