import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
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
# LIME
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
# SHAP
# ==============================
def explain_shap(image):
    print("\n🔍 Running SHAP...")
    img = np.reshape(image, (1, IMG_SIZE, IMG_SIZE, 3))
    masker = shap.maskers.Image("inpaint_telea", (IMG_SIZE, IMG_SIZE, 3))
    explainer = shap.Explainer(model, masker)
    shap_values = explainer(img, max_evals=100)
    shap.image_plot(shap_values)
# ==============================
# RUN
# ==============================
test_image = input("\nEnter full image path: ")
img = load_image(test_image)
if img is not None:
    print("\n=== PREDICTION ===")
    predict(img)
    print("\n=== LIME ===")
    explain_lime(test_image)
    print("\n=== SHAP ===")
    explain_shap(img)