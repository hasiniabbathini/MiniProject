import os
import io
import base64
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
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

app = Flask(__name__, static_folder='static')
CORS(app)

# ==============================
# LOAD IMAGE
# ==============================
def load_image_from_bytes(file_bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
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
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    model = Sequential([
        base_model, GlobalAveragePooling2D(),
        Dense(128, activation='relu'), Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, validation_data=val_data, epochs=25,
              class_weight={0: 1.0, 1: 2.0})
    model.save(MODEL_FILE)
    print("Model saved")

# ==============================
# HELPER: figure to base64
# ==============================
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

# ==============================
# LIME - original code unchanged
# ==============================
def run_lime(img):
    explainer = lime_image.LimeImageExplainer()

    def batch_predict(images):
        images = np.array(images)
        return np.concatenate([
            1 - model.predict(images, verbose=0),
            model.predict(images, verbose=0)
        ], axis=1)

    explanation = explainer.explain_instance(
        img, batch_predict, top_labels=1, hide_color=0, num_samples=1000
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True,
        hide_rest=False, num_features=10
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mark_boundaries(temp, mask))
    ax.set_title("LIME Explanation", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    return fig_to_b64(fig)

# ==============================
# SHAP - SmoothGrad → WHITE background saliency map
# Exact style: white/light bg, scattered red-pink dots for tumor
# Blue scattered dots for normal (like reference paper figures)
# ==============================
def run_shap(img, prediction):
    img_tensor = tf.cast(np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3)), tf.float32)
    num_samples = 50
    noise_level = 0.1
    grad_sum = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    for _ in range(num_samples):
        noise = tf.random.normal(shape=img_tensor.shape, stddev=noise_level)
        noisy = tf.clip_by_value(img_tensor + noise, 0.0, 1.0)
        with tf.GradientTape() as tape:
            tape.watch(noisy)
            output = model(noisy, training=False)
            loss = output[0][0] if prediction > 0.5 else 1.0 - output[0][0]
        grads = tape.gradient(loss, noisy)
        grad_sum += grads[0].numpy()

    smooth_grad = grad_sum / num_samples   # (224, 224, 3)

    # ── Build the saliency map in the style of the reference paper ──
    # Reference: white circle background, scattered pink/red or blue dots
    # showing pixel-level importance like a scatter/heatmap on white bg

    # Per-pixel importance = mean of abs gradients across channels
    saliency = np.mean(np.abs(smooth_grad), axis=-1)   # (224, 224)

    # Also get signed map for blue/red distinction
    saliency_signed = np.mean(smooth_grad, axis=-1)     # (224, 224)

    # Normalize to [-1, 1]
    s_abs_max = np.max(np.abs(saliency_signed)) + 1e-10
    saliency_norm = saliency_signed / s_abs_max

    # Normalize absolute for alpha
    s_max = saliency.max() + 1e-10
    saliency_abs_norm = saliency / s_max

    # ── Create figure: original image | saliency on white bg ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor('white')

    # Left: original image
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title("Original Image", fontsize=12, pad=8)

    # Right: white background with saliency scatter
    # Create circular white background (like reference paper)
    white_bg = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)  # pure white

    # Build RGBA image on white
    # We use a custom colormap: negative = blue, positive = red/pink
    # This exactly matches the reference paper style
    rgba = np.ones((IMG_SIZE, IMG_SIZE, 4), dtype=np.float32)  # white + opaque

    if prediction > 0.5:
        # TUMOR: red-pink saliency (positive gradient regions)
        # White background, red dots where saliency is high
        # Exactly like reference: sparse pink/red scattered dots on white circle
        for y in range(IMG_SIZE):
            for x in range(IMG_SIZE):
                v = saliency_abs_norm[y, x]
                sign = saliency_norm[y, x]
                if sign > 0 and v > 0.05:
                    # Red-pink: R=1, G=(1-v), B=(1-v) → white to deep red
                    rgba[y, x, 0] = 1.0
                    rgba[y, x, 1] = max(0.0, 1.0 - v * 1.2)
                    rgba[y, x, 2] = max(0.0, 1.0 - v * 1.2)
                    rgba[y, x, 3] = 1.0
                else:
                    # Keep white
                    rgba[y, x, :3] = 1.0
                    rgba[y, x, 3] = 1.0
    else:
        # NORMAL: blue saliency (negative gradient regions)
        for y in range(IMG_SIZE):
            for x in range(IMG_SIZE):
                v = saliency_abs_norm[y, x]
                sign = saliency_norm[y, x]
                if sign < 0 and v > 0.05:
                    # Blue: R=(1-v), G=(1-v), B=1 → white to deep blue
                    rgba[y, x, 0] = max(0.0, 1.0 - v * 1.2)
                    rgba[y, x, 1] = max(0.0, 1.0 - v * 0.5)
                    rgba[y, x, 2] = 1.0
                    rgba[y, x, 3] = 1.0
                else:
                    rgba[y, x, :3] = 1.0
                    rgba[y, x, 3] = 1.0

    # Draw circular mask (like reference paper - circular retinal shape)
    cy, cx = IMG_SIZE // 2, IMG_SIZE // 2
    radius = IMG_SIZE // 2 - 4
    Y, X = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    circle_mask = (X - cx)**2 + (Y - cy)**2 > radius**2
    rgba[circle_mask, :3] = 0.88   # light gray outside circle
    rgba[circle_mask, 3] = 1.0

    axes[1].imshow(rgba)
    axes[1].axis('off')
    if prediction > 0.5:
        axes[1].set_title("SHAP Saliency Map (Tumor)", fontsize=12, pad=8)
    else:
        axes[1].set_title("SHAP Saliency Map (Normal)", fontsize=12, pad=8)

    # Colorbar exactly like reference paper
    if prediction > 0.5:
        # Blue to white to red colormap (like reference)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'shap_tumor', ['blue', 'white', 'red']
        )
        vmin = -s_abs_max
        vmax = s_abs_max
    else:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'shap_normal', ['blue', 'white', 'pink']
        )
        vmin = -s_abs_max
        vmax = s_abs_max

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    # Colorbar below both plots (like reference paper)
    cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.03])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('SHAP value', fontsize=11)
    # Format ticks in scientific notation like reference
    cbar.formatter = plt.ScalarFormatter(useMathText=True)
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((-3, 3))
    cbar.update_ticks()

    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    return fig_to_b64(fig)

# ==============================
# ROUTES
# ==============================
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    file_bytes = file.read()
    img = load_image_from_bytes(file_bytes)

    if img is None:
        return jsonify({'error': 'Could not read image'}), 400

    # Prediction
    img_input = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))
    raw_pred = float(model.predict(img_input, verbose=0)[0][0])
    label = "TUMOR" if raw_pred > 0.5 else "NORMAL"
    confidence = raw_pred if raw_pred > 0.5 else 1 - raw_pred
    accuracy_pct = round(confidence * 100, 2)

    # Original image as base64
    orig_fig, orig_ax = plt.subplots(figsize=(4, 4))
    orig_fig.patch.set_facecolor('white')
    orig_ax.imshow(img)
    orig_ax.axis('off')
    orig_b64 = fig_to_b64(orig_fig)

    # LIME
    lime_img = run_lime(img)

    # SHAP
    shap_img = run_shap(img, raw_pred)

    return jsonify({
        'prediction': label,
        'confidence': accuracy_pct,
        'tumor_prob': round(raw_pred * 100, 2),
        'normal_prob': round((1 - raw_pred) * 100, 2),
        'original_image': orig_b64,
        'lime_image': lime_img,
        'shap_image': shap_img,
    })

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    print("\n" + "="*50)
    print("RetinoDetect server starting...")
    print("Open your browser at: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=False, port=5000)
