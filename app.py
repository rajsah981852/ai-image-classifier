import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import time

st.set_page_config(page_title="Cat-Dog Tool", layout="centered")

st.title("🐱🐶 AI Data Annotation + Training + Prediction")

DATA_DIR = "dataset"
CAT_DIR = os.path.join(DATA_DIR, "cat")
DOG_DIR = os.path.join(DATA_DIR, "dog")

os.makedirs(CAT_DIR, exist_ok=True)
os.makedirs(DOG_DIR, exist_ok=True)

menu = st.sidebar.selectbox("Menu", ["Annotate Data", "Train Model", "Predict"])

# -----------------------
# 1. DATA ANNOTATION
# -----------------------
if menu == "Annotate Data":
    st.header("📤 Upload & Label Images")

    label = st.radio("Select Label", ["cat", "dog"])

    uploaded_files = st.file_uploader(
        "Upload images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        key=label  # 🔥 FIX
    )

    if uploaded_files:
        save_path = CAT_DIR if label == "cat" else DOG_DIR

        for file in uploaded_files:
            file_path = os.path.join(save_path, file.name)

            if not os.path.exists(file_path):
                img = Image.open(file).convert("RGB")
                img.save(file_path)
                st.image(img, width=150)
            else:
                st.warning(f"{file.name} already exists, skipped")

        st.success(f"{len(uploaded_files)} images saved to {label}")

# -----------------------
# 2. TRAIN MODEL
# -----------------------
if menu == "Train Model":
    st.header("🧠 Train Your Model")

    # dataset check
    if len(os.listdir(CAT_DIR)) < 5 or len(os.listdir(DOG_DIR)) < 5:
        st.warning("⚠️ Add at least 5 images in each class before training")
        st.stop()

    if st.button("Start Training"):
        with st.spinner("Training in progress... ⏳"):

            img_size = (128, 128)

            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2,
                horizontal_flip=True,
                rotation_range=10
            )

            train_data = datagen.flow_from_directory(
                DATA_DIR,
                target_size=img_size,
                batch_size=8,
                class_mode='binary',
                subset='training'
            )

            val_data = datagen.flow_from_directory(
                DATA_DIR,
                target_size=img_size,
                batch_size=8,
                class_mode='binary',
                subset='validation'
            )

            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(128,128,3)),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            model.fit(train_data, validation_data=val_data, epochs=5, verbose=1)

            # save with timestamp (fix permission error)
            filename = f"cat_dog_model_{int(time.time())}.keras"
            model.save(filename)

        st.success(f"✅ Model saved as {filename}")

# -----------------------
# 3. PREDICTION
# -----------------------
if menu == "Predict":
    st.header("🔍 Test Your Model")

    # get latest model automatically
    model_files = [f for f in os.listdir() if f.endswith(".keras")]

    if model_files:
        latest_model = max(model_files, key=os.path.getctime)

        @st.cache_resource
        def load_model():
            return tf.keras.models.load_model(latest_model)

        model = load_model()

        st.info(f"Using model: {latest_model}")

        file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

        if file:
            img = Image.open(file)
            st.image(img, width=200)

            img = img.resize((128, 128))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)[0][0]

            if pred > 0.5:
                st.success("🐶 DOG")
            else:
                st.success("🐱 CAT")

    else:
        st.warning("Train model first!")