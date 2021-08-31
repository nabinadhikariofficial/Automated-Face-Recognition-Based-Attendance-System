from retinaface import RetinaFace
from matplotlib import pyplot as plt
from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
import cv2
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from sklearn.preprocessing import Normalizer

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

basedir = os.path.abspath(os.path.dirname(__file__))
modeldir = os.path.normpath(
    os.getcwd() + os.sep + os.pardir)+"\\Notebook_Scripts_Data\\model\\"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'key'


def face_detection(image_loc):
    faces = RetinaFace.detect_faces(img_path=image_loc)
    image = cv2.imread(image_loc)
    for face in faces.items():
        data = face[1]["facial_area"]
        # formula = (y1:y2+1 ,x1:x2+1)
        crop = image[data[1]:data[3]+1, data[0]:data[2]+1]
        location1 = basedir + '\\static\\img\\faces\\instant\\' + \
            str(face[0]) + '.jpg'
        location2 = basedir + '\\static\\img\\faces\\backup\\' + \
            str(face[0]) + '_'+str(int(time.time())) + '.jpg'
        cv2.imwrite(location1, crop)
        cv2.imwrite(location2, crop)
    return faces


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
    return result


host_add = '0.0.0.0'
port_add = 5000


@app.route('/')
def index():
    return render_template('login.html')

# webpage where user can provide image


@app.route('/detectfaces', methods=['GET', 'POST'])
def detectfaces():
    if request.method == 'POST':
        file = request.files['file']
        if 'file' not in request.files:
            message = "Please Select a Image first"
        elif file.filename == '':
            message = "Please Select a Image first"
        else:
            message = "Image accepted"
            filename = secure_filename("image_"+str(int(time.time()))+".jpg")
            file.save(os.path.join(
                basedir, app.config['UPLOAD_FOLDER'], filename))
            filename_full = basedir + "\\uploads\\" + filename
            info = face_detection(filename_full)
        context = {'message': message, 'image_info': info,
                   'img_time': str(int(time.time()))}
        return render_template('detectfaces.html', context=context, len=len(info), zip=zip)
    else:
        return render_template('detectfaces.html', context={}, len=0, zip=zip)


@app.route('/takeattendance', methods=['GET', 'POST'])
def takeattendance():
    if request.method == 'POST':
        file = request.files['file']
        if 'file' not in request.files:
            message = "Please Select a Image first"
        elif file.filename == '':
            message = "Please Select a Image first"
        else:
            message = "Image accepted"
            filename = secure_filename("image_"+str(int(time.time()))+".jpg")
            file.save(os.path.join(
                basedir, app.config['UPLOAD_FOLDER'], filename))
            filename_full = basedir + "\\uploads\\" + filename
            info = face_detection(filename_full)
            result = get_face_encodings(len(info))
            print(result)
        context = {'message': message, 'image_info': info,
                   'img_time': str(int(time.time()))}
        return render_template('takeattendance.html', context=context, len=len(info), zip=zip)
    else:
        return render_template('takeattendance.html', context={}, len=0, zip=zip)


# Running the app
if __name__ == "__main__":
    app.run(host=host_add, port=port_add, debug=True)
    app.jinja_env.filters['zip'] = zip
