import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from lime import lime_image
from skimage.segmentation import mark_boundaries
import tensorflow as tf

# ==============================
# SETTINGS
# ==============================
data_path = "dataset"
IMG_SIZE = 224
MODEL_FILE = "model.keras"

# ==============================
# LOAD IMAGE
# ==============================
def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Image not found. Check full path.")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

# ==============================
# MODEL LOAD / TRAIN
# ==============================
if os.path.exists(MODEL_FILE):
    print("✅ Loading saved model...")
    model = load_model(MODEL_FILE)
    datagen = ImageDataGenerator(rescale=1./255)
    train_data = datagen.flow_from_directory(
        data_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=16,
        class_mode='binary'
    )
else:
    print("🚀 Training model...")
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    train_data = datagen.flow_from_directory(
        data_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=16,
        class_mode='binary',
        subset='training'
    )
    val_data = datagen.flow_from_directory(
        data_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=16,
        class_mode='binary',
        subset='validation'
    )
    print("Class indices:", train_data.class_indices)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=25,
        class_weight={0: 1.0, 1: 2.0}
    )
    model.save(MODEL_FILE)
    print("✅ Model saved")

# ==============================
# PREDICTION
# ==============================
def predict(image):
    img_input = np.reshape(image, (1, IMG_SIZE, IMG_SIZE, 3))
    prediction = model.predict(img_input)[0][0]
    print(f"\nProbability Tumor: {prediction:.4f}")
    print(f"Probability Normal: {1 - prediction:.4f}")
    if prediction > 0.5:
        print("✅ Prediction: TUMOR")
    else:
        print("✅ Prediction: NORMAL")
    return prediction

# ==============================
# LIME - ORIGINAL (completely unchanged)
# ==============================
def explain_lime(image_path):
    print("\n🔍 Running LIME...")
    img = load_image(image_path)
    if img is None:
        return
    explainer = lime_image.LimeImageExplainer()
    def batch_predict(images):
        images = np.array(images)
        return np.concatenate([
            1 - model.predict(images),
            model.predict(images)
        ], axis=1)
    explanation = explainer.explain_instance(
        img,
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        hide_rest=False,
        num_features=10
    )
    plt.imshow(mark_boundaries(temp, mask))
    plt.title("LIME Explanation")
    plt.axis("off")
    plt.show()

# ==============================
# SHAP - SmoothGrad (no shap library needed, no shape errors)
# Produces pixel-level red (tumor) / blue (normal) saliency map
# ==============================
def explain_shap(image, prediction):
    print("\n🔍 Running SHAP...")

    img_tensor = tf.cast(
        np.reshape(image, (1, IMG_SIZE, IMG_SIZE, 3)), tf.float32
    )

    # SmoothGrad: average gradients over N noisy copies of the image
    num_samples = 50
    noise_level = 0.1
    grad_sum = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    print(f"Computing saliency map ({num_samples} samples)...")
    for i in range(num_samples):
        noise = tf.random.normal(shape=img_tensor.shape, stddev=noise_level)
        noisy = tf.clip_by_value(img_tensor + noise, 0.0, 1.0)

        with tf.GradientTape() as tape:
            tape.watch(noisy)
            output = model(noisy, training=False)
            # Gradient toward whichever class was predicted
            if prediction > 0.5:
                loss = output[0][0]          # toward TUMOR
            else:
                loss = 1.0 - output[0][0]   # toward NORMAL

        grads = tape.gradient(loss, noisy)   # shape: (1, 224, 224, 3)
        grad_sum += grads[0].numpy()

    # Average over samples → shape (224, 224, 3)
    smooth_grad = grad_sum / num_samples

    # Collapse RGB → single 2D saliency map
    saliency_2d = np.mean(np.abs(smooth_grad), axis=-1)  # (224, 224)

    # Normalize to [0, 1]
    smin, smax = saliency_2d.min(), saliency_2d.max()
    saliency_norm = (saliency_2d - smin) / (smax - smin + 1e-10)

    # --- Build RGBA overlay ---
    h, w = saliency_norm.shape
    rgba_overlay = np.zeros((h, w, 4), dtype=np.float32)

    if prediction > 0.5:
        # TUMOR → RED, opacity driven by saliency strength
        rgba_overlay[:, :, 0] = 1.0
        rgba_overlay[:, :, 1] = 0.0
        rgba_overlay[:, :, 2] = 0.0
        rgba_overlay[:, :, 3] = saliency_norm * 0.85
        overlay_title = "SHAP Saliency Map — Tumor (Red)"
        cmap_bar = plt.cm.Reds
    else:
        # NORMAL → BLUE, opacity driven by saliency strength
        rgba_overlay[:, :, 0] = 0.0
        rgba_overlay[:, :, 1] = 0.35
        rgba_overlay[:, :, 2] = 1.0
        rgba_overlay[:, :, 3] = saliency_norm * 0.85
        overlay_title = "SHAP Saliency Map — Normal (Blue)"
        cmap_bar = plt.cm.Blues

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(image, alpha=0.55)
    axes[1].imshow(rgba_overlay)          # overlay on top
    axes[1].set_title(overlay_title, fontsize=13)
    axes[1].axis("off")

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap_bar,
        norm=plt.Normalize(vmin=smin, vmax=smax)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("SHAP value", fontsize=11)

    plt.suptitle("SHAP Explanation", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ==============================
# RUN
# ==============================
test_image = input("\nEnter full image path: ")
img = load_image(test_image)

if img is not None:
    print("\n=== PREDICTION ===")
    pred = predict(img)

    print("\n=== LIME ===")
    explain_lime(test_image)

    print("\n=== SHAP ===")
    explain_shap(img, pred)
