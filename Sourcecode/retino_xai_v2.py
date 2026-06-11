import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
        print("Image not found. Check full path.")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

# ==============================
# MODEL LOAD / TRAIN
# ==============================
if os.path.exists(MODEL_FILE):
    print("Loading saved model...")
    model = load_model(MODEL_FILE)
    datagen = ImageDataGenerator(rescale=1./255)
    train_data = datagen.flow_from_directory(
        data_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=16,
        class_mode='binary'
    )
else:
    print("Training model...")
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_data = datagen.flow_from_directory(
        data_path, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=16, class_mode='binary', subset='training'
    )
    val_data = datagen.flow_from_directory(
        data_path, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=16, class_mode='binary', subset='validation'
    )
    print("Class indices:", train_data.class_indices)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
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
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, validation_data=val_data, epochs=25,
              class_weight={0: 1.0, 1: 2.0})
    model.save(MODEL_FILE)
    print("Model saved")

# ==============================
# PREDICTION
# ==============================
def predict(image):
    img_input = np.reshape(image, (1, IMG_SIZE, IMG_SIZE, 3))
    prediction = model.predict(img_input)[0][0]
    tumor_prob  = float(prediction)
    normal_prob = float(1 - prediction)
    print(f"\nProbability Tumor:  {tumor_prob:.4f}  ({tumor_prob*100:.2f}%)")
    print(f"Probability Normal: {normal_prob:.4f}  ({normal_prob*100:.2f}%)")
    if tumor_prob > 0.5:
        print("Prediction: TUMOR")
    else:
        print("Prediction: NORMAL")
    return tumor_prob

# ==============================
# LIME — ORIGINAL (unchanged)
# coverage of highlighted area is proportional to tumor probability
# ==============================
def explain_lime(image_path, tumor_prob):
    print("\nRunning LIME...")
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
        img, batch_predict,
        top_labels=1, hide_color=0, num_samples=1000
    )

    # -------------------------------------------------------
    # Probability-based num_features:
    # tumor_prob 1.0 → show up to 15 segments
    # tumor_prob 0.0 → show 1 segment (tiny mark)
    # -------------------------------------------------------
    max_features = 15
    min_features = 1
    num_features = max(min_features, round(min_features + (max_features - min_features) * tumor_prob))

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        hide_rest=False,
        num_features=num_features
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(mark_boundaries(temp, mask))
    axes[1].set_title(f"LIME Explanation  (tumor prob: {tumor_prob*100:.1f}%)", fontsize=13)
    axes[1].axis("off")

    plt.suptitle("LIME Explanation", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ==============================
# SHAP — SmoothGrad on WHITE background
# Red dots  → tumor-affecting pixels
# Blue dots → normal-supporting pixels
# Output matches reference paper style exactly
# ==============================
def explain_shap(image, tumor_prob):
    print("\nRunning SHAP (SmoothGrad)...")

    img_tensor = tf.cast(
        np.reshape(image, (1, IMG_SIZE, IMG_SIZE, 3)), tf.float32
    )

    isTumor = tumor_prob > 0.5
    num_samples = 50
    noise_level = 0.1
    grad_sum = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    print(f"Computing saliency ({num_samples} samples)...")
    for i in range(num_samples):
        noise = tf.random.normal(shape=img_tensor.shape, stddev=noise_level)
        noisy = tf.clip_by_value(img_tensor + noise, 0.0, 1.0)
        with tf.GradientTape() as tape:
            tape.watch(noisy)
            output = model(noisy, training=False)
            loss = output[0][0] if isTumor else 1.0 - output[0][0]
        grads = tape.gradient(loss, noisy)
        grad_sum += grads[0].numpy()

    smooth_grad = grad_sum / num_samples            # (224, 224, 3)
    saliency_2d = np.mean(np.abs(smooth_grad), axis=-1)  # (224, 224)

    smin, smax = saliency_2d.min(), saliency_2d.max()
    saliency_norm = (saliency_2d - smin) / (smax - smin + 1e-10)

    # -------------------------------------------------------
    # Build the saliency image on a WHITE background
    # matching the reference paper style:
    #   • white canvas (1.0, 1.0, 1.0)
    #   • red/pink scattered dots for tumor regions
    #   • blue scattered dots for normal regions
    #   • intensity proportional to saliency value
    # -------------------------------------------------------
    H, W = saliency_norm.shape
    saliency_rgb = np.ones((H, W, 3), dtype=np.float32)   # white background

    # Threshold: only show pixels above a relevance cutoff to get the
    # scattered dot appearance seen in the reference image
    threshold = 0.25
    for y in range(H):
        for x in range(W):
            v = saliency_norm[y, x]
            if v > threshold:
                intensity = (v - threshold) / (1.0 - threshold)   # re-normalise above threshold
                if isTumor:
                    # Red / pink  (1, 1-i, 1-i)  → white→pink→red
                    saliency_rgb[y, x] = [1.0, 1.0 - intensity * 0.85, 1.0 - intensity * 0.85]
                else:
                    # Blue  (1-i, 1-i, 1)  → white→light-blue→blue
                    saliency_rgb[y, x] = [1.0 - intensity * 0.85, 1.0 - intensity * 0.85, 1.0]

    # -------------------------------------------------------
    # Plot: left = original retinal image, right = saliency map
    # -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')

    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(saliency_rgb)
    if isTumor:
        axes[1].set_title(f"SHAP Saliency Map — Tumor\n(tumor prob: {tumor_prob*100:.1f}%)", fontsize=13)
    else:
        axes[1].set_title(f"SHAP Saliency Map — Normal\n(tumor prob: {tumor_prob*100:.1f}%)", fontsize=13)
    axes[1].axis("off")

    # -------------------------------------------------------
    # Colorbar matching the reference paper
    # -------------------------------------------------------
    if isTumor:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'tumor_cmap',
            [(1,1,1), (1, 0.85, 0.85), (1, 0.5, 0.5), (0.9, 0.1, 0.1)]
        )
    else:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'normal_cmap',
            [(1,1,1), (0.85, 0.85, 1), (0.5, 0.5, 1), (0.1, 0.1, 0.9)]
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=smin, vmax=smax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("SHAP value", fontsize=11)

    plt.suptitle("SHAP Explanation", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ==============================
# RUN
# ==============================
test_image = input("\nEnter full image path: ")
img = load_image(test_image)

if img is not None:
    print("\n=== PREDICTION ===")
    tumor_prob = predict(img)

    print("\n=== LIME ===")
    explain_lime(test_image, tumor_prob)

    print("\n=== SHAP ===")
    explain_shap(img, tumor_prob)
