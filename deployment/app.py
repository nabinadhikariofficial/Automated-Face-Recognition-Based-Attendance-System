from retinaface import RetinaFace
from matplotlib import pyplot as plt
from flask import Flask, request, render_template, jsonify, Markup, session, redirect, url_for, Response
import os
from werkzeug.utils import secure_filename
import cv2
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from sklearn.preprocessing import Normalizer
import datetime


UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

basedir = os.path.abspath(os.path.dirname(__file__))
maindir = basedir[:-11]

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
    model = load_model(
        maindir+"Notebook_Scripts_Data\\model\\facenet_keras.h5")
    model_svc = pickle.load(
        open(maindir+'Notebook_Scripts_Data\\model\\20210831-184738_svc.pk', 'rb'))
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
def Index():
    return render_template('login.html')

# webpage where user can provide image


@app.route('/DetectFaces', methods=['GET', 'POST'])
def DetectFaces():
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
        return render_template('DetectFaces.html', context=context, len=len(info), zip=zip)
    else:
        return render_template('DetectFaces.html', context={}, len=0, zip=zip)


@app.route('/TakeAttendance', methods=['GET', 'POST'])
def TakeAttendance():
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
        return render_template('TakeAttendance.html', context=context, len=len(info), zip=zip)
    else:
        return render_template('TakeAttendance.html', context={}, len=0, zip=zip)


global capture
capture = 0


@app.route('/CameraAttendance', methods=['GET', 'POST'])
def CameraAttendance():
    global capture
    if request.method == 'POST':
        if request.form.get("capture") == 'Capture':
            global capture
            capture = 1
    return render_template('CameraAttendance.html')


def live_video():
    global capture
    cascPath = "./haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    camera = cv2.VideoCapture(0)

    while True:

        success, frame = camera.read()  # read the camera frame
        try:
            os.mkdir('./capture')
        except OSError as error:
            pass

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if not success:
            break
        else:
            if(capture):
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(
                    ['capture', "capture_{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(p, frame)
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            except Exception as e:
                pass
    camera.release
    cv2.destroyAllWindows()


@app.route('/capture_feed')
def capture_feed():
    return Response(live_video(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/AttendanceDetails', methods=['GET', 'POST'])
def AttendanceDetails():
    return render_template('AttendanceDetails.html')


# Running the app
if __name__ == "__main__":
    app.run(host=host_add, port=port_add, debug=True)
    app.jinja_env.filters['zip'] = zip
