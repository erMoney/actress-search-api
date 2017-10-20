import base64

import numpy as np
import pandas as pd
from cv2 import cv2
from flask import Flask, jsonify, json, render_template, request, Response
from keras.models import load_model

app = Flask(__name__)

# Load Cascade
FACE_DETECT_CASCADE = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
EYES_DETECT_CASCADE = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

# Load Actress List
ACTRESS_LIST = [name for i, name in enumerate(pd.read_csv('actress_list.txt').name)]


class NoFaceDetectError(Exception):
    status_code = 404

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


def resize(image):
    return cv2.resize(image, (64, 64))


def detect_faces(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_rect_list = FACE_DETECT_CASCADE.detectMultiScale(gray_img, scaleFactor=1.03, minNeighbors=5, minSize=(10, 10))
    face_img_list = []
    for i, face_rect in enumerate(face_rect_list):
        face_img = resize(img[face_rect[1]:face_rect[1] + face_rect[3], face_rect[0]:face_rect[0] + face_rect[2]])
        # if has_two_eyes(face_img):
        #     face_img_list.append(face_img)
        face_img_list.append(face_img)
    return face_img_list


def has_two_eyes(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = EYES_DETECT_CASCADE.detectMultiScale(image_gray, scaleFactor=1.03, minNeighbors=5, minSize=(10, 10))
    return len(eyes) == 2


def recognize_face_name(img):
    face_img_list = detect_faces(img)
    if len(face_img_list) == 0:
        raise NoFaceDetectError('Face is not detected.', status_code=404)
    face_img = face_img_list[0]
    x = np.array([face_img]).astype('float') / 256
    # Load Model
    model = load_model('model.h5')
    idx = np.argmax(model.predict(x, batch_size=32)[0], axis=-1)
    return ACTRESS_LIST[idx]


def read_base64_img(base64_img):
    img = base64.b64decode(base64_img)
    return cv2.imdecode(np.fromstring(img, np.uint8), cv2.COLOR_RGB2BGR)


@app.errorhandler(NoFaceDetectError)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route('/face:recognition', methods=['POST'])
def recognize():
    data = json.loads(request.data)
    img = read_base64_img(data['image'])
    name = recognize_face_name(img)
    return jsonify({'face': {'name': name}})


if __name__ == '__main__':
    app.debug=True
    app.run(host='0.0.0.0')
