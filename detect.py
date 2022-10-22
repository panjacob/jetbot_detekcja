import os

import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import traitlets
from IPython.display import display
import ipywidgets.widgets as widgets

try:
    from jetbot import Camera
    from jetbot import bgr8_to_jpeg
    from jetbot import Robot
except Exception as e:
    print(e)
import time

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
        print(pred_value[0])

        cv2.waitKey(1)

    cam.release()
    cv2.destroyAllWindows()


def update(change):
    global blocked_slider, robot
    x = change['new']
    x = preprocess_image(x)  # To wstawione moje
    y = model.predict(x)

    prob_blocked = float(y.flatten()[0])
    print(prob_blocked)

    blocked_slider.value = prob_blocked

    if prob_blocked < 0.5:
        robot.forward(speed_slider.value)
    else:
        robot.left(speed_slider.value)

    time.sleep(0.001)


def preprocess_image(frame):
    image = cv2.resize(frame, (50, 50))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_arr = img_to_array(image)
    img_arr = img_arr / 255.
    np_image = np.expand_dims(img_arr, axis=0)
    return np_image


if __name__ == "__main__":
    model = load_model(MODEL_CHECKPOINT_PATH)
    # classify_image_opencv()

    camera = Camera.instance(width=50, height=50)
    image = widgets.Image(format='jpeg', width=50, height=50)
    blocked_slider = widgets.FloatSlider(description='blocked', min=0.0, max=1.0, orientation='vertical')
    speed_slider = widgets.FloatSlider(description='speed', min=0.0, max=0.5, value=0.0, step=0.01,
                                       orientation='horizontal')

    camera_link = traitlets.dlink((camera, 'value'), (image, 'value'), transform=bgr8_to_jpeg)

    display(widgets.VBox([widgets.HBox([image, blocked_slider]), speed_slider]))
    robot = Robot()
    update({'new': camera.value})  # we call the function once to initialize
