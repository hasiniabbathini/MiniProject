# Run this ONCE to convert your model.keras to TF.js format
# Install: pip install tensorflowjs

import tensorflowjs as tfjs
from tensorflow.keras.models import load_model

print("Loading model...")
model = load_model("model.keras")

print("Converting to TF.js format...")
tfjs.converters.save_keras_model(model, "./tfjs_model")

print("Done! Files saved to ./tfjs_model/")
print("Copy model.json and the .bin file(s) next to RetinoDetect.html")
print("Then run: python -m http.server 8080")
print("Open: http://localhost:8080/RetinoDetect.html")
