import uuid

import cv2
import dlib
import os

import imutils
import openface

IMAGE_HEIGHT = 300
FACE_IMAGE_SIZE = 64
ROTATE_ANGLES = [0, -22.5, 22.5, -45, 45]

DIRECTORY = os.path.dirname(__file__)
FACE_DETECT_CASCADE = cv2.CascadeClassifier(os.path.join(DIRECTORY, '../models/haarcascades/haarcascade_frontalface_default.xml'))
EYES_DETECT_CASCADE = cv2.CascadeClassifier(os.path.join(DIRECTORY, '../models/haarcascades/haarcascade_eye.xml'))

PREDICTOR_MODEL     = os.path.join(DIRECTORY, '../models/shape_predictor_68_face_landmarks.dat')
DLIB_FACE_DETECTOR  = dlib.get_frontal_face_detector()
DLIB_FACE_PRODICTOR = dlib.shape_predictor(PREDICTOR_MODEL)

class Face:
    def __init__(self, cvImg):
        self.cvImg = cvImg

    def write(self, filename):
        cv2.imwrite(filename, self.cvImg)

    def data(self):
        return self.cvImg


def pre_resize(img, height=IMAGE_HEIGHT):
    size = tuple([img.shape[1], img.shape[0]])
    x = int(round(float(height / float(size[1]) * float(size[0]))))
    y = height
    return cv2.resize(img, (x, y))


def resize(img):
    return cv2.resize(img, (FACE_IMAGE_SIZE, FACE_IMAGE_SIZE))

def rotate(img, angle):
    return imutils.rotate_bound(img, angle)

def has_two_eyes(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = EYES_DETECT_CASCADE.detectMultiScale(gray_img, scaleFactor=1.03, minNeighbors=5, minSize=(10, 10))
    return len(eyes) == 2


def detect_faces(filename):
    img = cv2.imread(filename)
    detect_faces(img)


def crop_image(img, top, left, right, bottom):
    return img[left:right, top:bottom]


def detect_faces(img, module='dlib'):
    for angle in ROTATE_ANGLES:
        pre_size_img = pre_resize(img)
        rotate_img   = rotate(pre_size_img, angle)
        gray_img     = cv2.cvtColor(rotate_img, cv2.COLOR_BGR2GRAY)

        faces = []
        if module == 'dlib':
            rects, _, _ = DLIB_FACE_DETECTOR.run(gray_img, 1, -0.1)
            for d in rects:
                face_aligner = openface.AlignDlib(PREDICTOR_MODEL)
                face_img = face_aligner.align(300, rotate_img, d,
                                                 landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
                face = Face(resize(face_img))
                faces.append(face)
        else:
            rects = FACE_DETECT_CASCADE.detectMultiScale(gray_img, scaleFactor=1.03, minNeighbors=5, minSize=(10, 10))
            for d in rects:
                face_img = crop_image(rotate_img, d[0], d[1], d[1] + d[3], d[0] + d[2])
                face = Face(resize(face_img))
                if has_two_eyes(face_img):
                    faces.append(face)
        if len(faces) > 0:
            return faces
    return []


def detect_faces_and_save(filename, dir, module='dlib'):
    faces = detect_faces(cv2.imread(filename), module)
    for face in faces:
        face.write(os.path.join(dir, str(uuid.uuid4()) + ".jpg"))
    return len(faces)
