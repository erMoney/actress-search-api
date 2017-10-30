import base64

import numpy as np
import pandas as pd
from cv2 import cv2
from flask import Flask, jsonify, json, render_template, request, Response
from keras.models import load_model
import tensorflow as tf
import os

from lib.face_detect import detect_faces

app = Flask(__name__)

# Load Actress List
ACTRESS_LIST = [name for i, name in enumerate(pd.read_csv('actress_list.txt').name)]

# Load model
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(DIRECTORY, 'models/model.h5'))

# https://github.com/fchollet/keras/issues/2397
graph = tf.get_default_graph()

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

def recognize_face_name(img):
    # 別スレッドでmodelをロードした場合
    # https://github.com/fchollet/keras/issues/2397
    global graph
    with graph.as_default():
        face_img_list = detect_faces(img)
        if len(face_img_list) == 0:
            raise NoFaceDetectError('Face is not detected.', status_code=404)
        face_img = face_img_list[0]
        x = np.array([face_img]).astype('float') / 256
        # Load Model
        predict = model.predict(x)[0]
        candidates = []
        for idx in np.argsort(predict)[::-1]:
            score = predict[idx].item()
            candidates.append({'name': ACTRESS_LIST[idx], 'score': score})
        return candidates[0]['name'], candidates


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
    name, candidates = recognize_face_name(img)
    body = {'face': {'name': name},'candidates':candidates}
    return jsonify(body)


if __name__ == '__main__':
    app.debug=True
    app.run(host='0.0.0.0')
