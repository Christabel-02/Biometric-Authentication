import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
IMG_SIZE = 128
DATA_DIR = "mmu-iris-dataset/MMU-Iris-Database/"

def load_dataset():
    X, y = [], []
    for person_id in sorted(os.listdir(DATA_DIR)):
        person_path = os.path.join(DATA_DIR, person_id)
        if not os.path.isdir(person_path): continue

        for eye in ['left', 'right']:
            eye_path = os.path.join(person_path, eye)
            if not os.path.exists(eye_path): continue

            for img_name in os.listdir(eye_path):
                if not img_name.endswith('.bmp'): continue
                img_path = os.path.join(eye_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(int(person_id))  # label is folder name

    X = np.array(X, dtype="float32") / 255.0
    y = np.array(y)
    X = np.expand_dims(X, -1)
    y_cat = to_categorical(y)
    return X, y_cat, y

def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Streamlit UI
st.title("👁️ Iris Recognition System with MMU Dataset")

with st.spinner("Loading dataset..."):
    X, y_cat, y = load_dataset()

if X.size == 0:
    st.error("❌ Dataset not loaded. Check file paths and image formats.")
else:
    st.success(f"✅ Loaded {len(X)} images.")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    # Data Augmentation
    st.sidebar.title("🧪 Augmentation")
    apply_aug = st.sidebar.checkbox("Apply Rotation (±15°)")
    datagen = ImageDataGenerator(rotation_range=15 if apply_aug else 0)
    datagen.fit(X_train)

    
    model = build_model(num_classes=y_cat.shape[1])

    if st.button("Train Model"):
        with st.spinner("Training..."):
            model.fit(datagen.flow(X_train, y_train, batch_size=32),
                      epochs=10,
                      validation_data=(X_test, y_test),
                      verbose=0)
        acc = model.evaluate(X_test, y_test, verbose=0)[1]
        st.success(f"✅ Model trained. Accuracy: {acc:.2%}")

    st.markdown("### 🔍 Test on a Random Image")
    if st.button("Pick and Predict"):
        idx = np.random.randint(len(X_test))
        image = X_test[idx]
        true_class = np.argmax(y_test[idx])
        pred_class = np.argmax(model.predict(image[np.newaxis, ...]))
        st.image(image.squeeze(), caption=f"True: {true_class} | Predicted: {pred_class}", width=200)
