from retinaface import RetinaFace
from matplotlib import pyplot as plt
from flask import Flask, request, render_template, session, redirect, url_for, Response
import hashlib
import mysql.connector
import os
from werkzeug.utils import secure_filename
import cv2
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import pandas as pd
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
        maindir+"\\Notebook_Scripts_Data\\model\\facenet_keras.h5")
    model_svc = pickle.load(
        open(maindir+'\\Notebook_Scripts_Data\\model\\20210831-184738_svc.pk', 'rb'))
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


# database connection details below
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="attendance"
)

# host address and port
host_add = '0.0.0.0'
port_add = 5000

# making cursor
cursor = mydb.cursor(dictionary=True)



@app.route('/', methods=['GET', 'POST'])
def Index():
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if 'loggedin' in session:
        return redirect(url_for('Profile'))
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        username=username.upper()
        password = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute(
            'SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password,))
        account = cursor.fetchone()
        if account:
            # Create session data, we can access this data in other routes
            # session.permanent = True
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            session['access']=account['access']
            # Redirect to profile page
            return redirect(url_for('Profile'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect Username or Password'
    return render_template('login.html',msg=msg)

@app.route('/Profile', methods=['GET', 'POST'])
def Profile():
    if 'loggedin' in session:
        user_data = pd.read_csv(maindir+"\\Notebook_Scripts_Data\\studentdetails.csv", index_col=0).T[session['username']].to_dict()
        return render_template('profile.html',user_data=user_data)
    return redirect(url_for('Index'))
# webpage where user can provide image

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    session.pop('access',None)
    # Redirect to login page
    return redirect(url_for('Index'))

@app.route('/DetectFaces', methods=['GET', 'POST'])
def DetectFaces():
    if 'loggedin' in session:
        if session['access']!='S':
            if request.method == 'POST':
                file = request.files['file']
                if (not file):
                    print("no file")
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
            return render_template('DetectFaces.html', context={}, len=0, zip=zip)
    return redirect(url_for('Index'))

@app.route('/TakeAttendance', methods=['GET', 'POST'])
def TakeAttendance():
    if 'loggedin' in session:
        if session['access']!='S':
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
                    present = []
                    data = pd.read_csv(
                        maindir+"\\Notebook_Scripts_Data\\crnAndName.csv")
                    for i in data['CRN']:
                        count = 0
                        for j in result:
                            if i in j:
                                present.append('Present')
                                count = 1
                                break
                            else:
                                continue
                        if count == 0:
                            present.append("Absent")
                    data["Status"] = present
                    data_list = data.values.tolist()
                    title = (data.columns.values.tolist())
                    total = len(present)
                    present_no = len(result)
                    absent_no = total - present_no
                context = {'message': message, 'image_info': info,
                       'img_time': str(int(time.time()))}
                return render_template('TakeAttendance.html', context=context, len=len(info), tables=data_list, title=title, result=result, total=total, present=present_no, absent=absent_no)
            else:
                return render_template('TakeAttendance.html', context={}, len=0)
    return redirect(url_for('Index'))

global capture
capture = 0


@app.route('/CameraAttendance', methods=['GET', 'POST'])
def CameraAttendance():
    global capture
    if 'loggedin' in session:
        if session['access']!='S':
            if request.method == 'POST':
                if request.form.get("capture") == 'Capture':
                    global capture
                    capture = 1
            return render_template('CameraAttendance.html')
    return redirect(url_for('Index'))

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
    if 'loggedin' in session:
        return Response(live_video(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return redirect(url_for('Index'))


@app.route('/AttendanceDetails', methods=['GET', 'POST'])
def AttendanceDetails():
    if 'loggedin' in session:
        return render_template('AttendanceDetails.html')
    return redirect(url_for('Index'))


# Running the app
if __name__ == "__main__":
    app.run(host=host_add, port=port_add, debug=True)
    app.jinja_env.filters['zip'] = zip
