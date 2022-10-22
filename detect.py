import os

import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_PATH = os.path.join('model')
MODEL_CHECKPOINT_PATH = os.path.join('model', 'best_model.ckpt')


def generate_model():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1), padding='same'))
    # model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    # model.add(tf.keras.activations.relu(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    # model.add(tf.keras.activations.relu(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    # model.add(tf.keras.activations.relu(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation=tf.nn.relu))
    # model.add(tf.keras.activations.relu(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # model.summary()
    return model


def load_model(model_path):
    model = generate_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights(model_path)
    return model


def classify_image_opencv():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("window")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("window", frame)
        image = preprocess_image(frame)
        pred_value = model.predict(image)
        print(pred_value)

        cv2.waitKey(1)

    cam.release()
    cv2.destroyAllWindows()




def preprocess_image(frame):
    image = cv2.resize(frame, (50, 50))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_arr = img_to_array(image)
    img_arr = img_arr / 255.
    np_image = np.expand_dims(img_arr, axis=0)
    return np_image


if __name__ == "__main__":
    model = load_model(MODEL_CHECKPOINT_PATH)
    classify_image_opencv()

