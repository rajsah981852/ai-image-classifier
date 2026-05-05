AI Image Classifier

A simple AI-based image classification system that detects cats and dogs using TensorFlow and Streamlit. This project demonstrates an end-to-end machine learning pipeline including data annotation, model training, and prediction.

---

Features

* Upload and label images (Cat / Dog)
* Automatic dataset organization
* Train a deep learning model
* Predict whether an image is a cat or a dog
* Simple user interface built with Streamlit

---

Tech Stack

* Python
* TensorFlow / Keras
* Streamlit
* NumPy
* Pillow

---

Project Structure

```
ai-image-classifier/
│── app.py
│── train.py
│── requirements.txt
│── .gitignore
```

---

How to Run

1. Clone the repository

```
git clone https://github.com/your-username/ai-image-classifier.git
cd ai-image-classifier
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the application

```
streamlit run app.py
```

---

How It Works

1. Upload images and assign labels (cat or dog)
2. Images are stored in structured dataset folders
3. A convolutional neural network is trained on the dataset
4. A new image can be uploaded for prediction
5. The model outputs the predicted class

---

Important Notes

* The dataset is not included in the repository
* Users must upload their own images
* For better accuracy:

  * Use at least 50 images per class
  * Maintain a balanced dataset

---

Future Improvements

* Integration of pre-trained models such as MobileNet or ResNet
* Support for multi-class classification
* Visualization of training metrics
* Deployment as a web application

