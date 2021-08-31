from retinaface import RetinaFace
from matplotlib import pyplot as plt
from flask import Flask, request, render_template, jsonify, Markup, session, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
import time
from keras.models import load_model
import pickle
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import Normalizer

basedir = os.path.abspath(os.path.dirname(__file__))
modeldir = os.path.normpath(
    os.getcwd() + os.sep + os.pardir)+"\\Notebook_Scripts_Data\\model\\"

# Define image size
IMG_SIZE = 160


def process_image(image_path):
    """
    Takes an image file path and turns it into a Tensor.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    image = tf.expand_dims(image, axis=0)
    return image


def get_face_encodings(images):
    model = load_model(modeldir+'facenet_keras.h5')
    model_svc = pickle.load(open(modeldir+'20210831-184738_svc.pk', 'rb'))
    for image in range(images):
        image_path = basedir + "\\static\\img\\faces\\instant\\face_" + \
            str(image+1) + ".jpg"
        image_data = process_image(image_path)
        image_emb = model.predict(image_data)
        in_encode = Normalizer(norm='l2')
        image_emb_nom = in_encode.transform(image_emb)
        if image == 0:
            temp = image_emb_nom
        else:
            temp = np.vstack((temp, image_emb_nom))
    result = model_svc.predict(temp)
    print(result)


get_face_encodings(2)
