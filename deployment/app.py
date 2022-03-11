from retinaface import RetinaFace
from flask import Flask, request, render_template, session, redirect, url_for, Response,flash
import hashlib
import os
from werkzeug.utils import secure_filename
import cv2
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
import json

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

basedir = os.path.abspath(os.path.dirname(__file__))
maindir = basedir[:-11]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'key'


def face_detection(image_loc):
    print("Face detection Started....")
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
    print("Face have been detected....")
    return faces


# Define image size
IMG_SIZE = 160


def process_image(image_path):
    """
    Takes an image file path and turns it into a Tensor.
    """
    print("Image processing stared....")
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    image = tf.expand_dims(image, axis=0)
    print("Image processing finished....")
    return image


def get_face_encodings(images):
    print("loading models....")
    model = load_model(
        maindir+"\\Notebook_Scripts_Data\\model\\facenet_keras.h5")
    model_svc = pickle.load(
        open(maindir+'\\Notebook_Scripts_Data\\model\\20220223-210250_svc.pk', 'rb'))
    result_final=[]
    pred_final=[]
    for image in range(images):
        image_path = basedir + "\\static\\img\\faces\\instant\\face_" + \
            str(image+1) + ".jpg"
        image_data = process_image(image_path)
        print("128 embedding predict....")
        image_emb = model.predict(image_data)
        print("128 embedding done....")
        in_encode = Normalizer(norm='l2')
        image_emb_nom = in_encode.transform(image_emb)
        if image == 0:
            temp = image_emb_nom
        else:
            temp = np.vstack((temp, image_emb_nom))
    for out in temp:
        print("Checking result probabilty....")
        pred_prob=model_svc.predict_proba(out.reshape(1,-1))*100
        if (pred_prob.max()>10):
            result = model_svc.predict(out.reshape(1,-1))
            result_final.append(result[0])
        else:
            result_final.append('Unknown')
        pred_final.append(pred_prob.max())
    print("result calcultion finished....")
    print(result_final)
    print(pred_final)
    return result_final


# # database connection details below
# mydb = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     passwd="",
#     database="attendance"
# )

# host address and port
host_add = '0.0.0.0'
port_add = 5000

# # making cursor
# cursor = mydb.cursor(dictionary=True)



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
        account = pd.read_csv(maindir+"\\Notebook_Scripts_Data\\accounts.csv", index_col=0).T
        if username in account.columns:
            # fetch data
            account=account[username].to_dict()
            # Create session data, we can access this data in other routes
            # session.permanent = True
            session['loggedin'] = True
            session['username'] = username
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

def update_attendance(data):
    attendance_data = json.load(open(maindir+"\\Notebook_Scripts_Data\\data.json"))
    subject_selected=request.form["subject"]
    p_list=[]
    a_list=[]
    for data in data:
        if data[2]=='Present' :
            p_list.append(data[0])
            attendance_data['student'][data[0]][subject_selected]['Present']+=1
        else:
            a_list.append(data[0])
        attendance_data['student'][data[0]][subject_selected]['total']+=1
    temp={'date':datetime.now().strftime("%Y/%m/%d %H:%M:%S"),"present_list":p_list,"absent_list":a_list}
    attendance_data['attendance'][subject_selected].append(temp)
    with open(maindir+"\\Notebook_Scripts_Data\\data.json","w" )as f:
        f.write(json.dumps(attendance_data))
    return

capture_bool = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def attendance_processor(filename_full):
    message = "Image accepted"
    info = face_detection(filename_full)
    result = get_face_encodings(len(info))
    present = []
    data = pd.read_csv(maindir+"\\Notebook_Scripts_Data\\crnAndName.csv")
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
    present_no = len(result)-result.count('Unknown')
    absent_no = total - present_no
    print("updating attendance.....")
    update_attendance(data_list)
    context = {'message': message, 'image_info': info,'img_time': str(int(time.time()))}
    return context,info,data_list,title,result,total,present_no,absent_no

@app.route('/TakeAttendance', methods=['GET', 'POST'])
def TakeAttendance():
    if 'loggedin' in session:
        if session['access']!='S':
            if request.method == 'POST':
                file = request.files['file']
                if capture_bool:
                    print("Capture File have been accepted.....")
                    context,info,data_list,title,result,total,present_no,absent_no=attendance_processor(cap_path)
                    return render_template('TakeAttendance.html', context=context, len=len(info), tables=data_list, title=title, result=result, total=total, present=present_no, absent=absent_no,login=session['username'])
                elif file and allowed_file(file.filename):
                    filename = secure_filename("image_"+str(int(time.time()))+".jpg")
                    file.save(os.path.join(basedir, app.config['UPLOAD_FOLDER'], filename))
                    filename_full = basedir + "\\uploads\\" + filename
                    print("File have been accepted.....")
                    context,info,data_list,title,result,total,present_no,absent_no=attendance_processor(filename_full)
                    return render_template('TakeAttendance.html', context=context, len=len(info), tables=data_list, title=title, result=result, total=total, present=present_no, absent=absent_no,login=session['username'])
                return redirect(url_for('TakeAttendance'))
            else:
                return render_template('TakeAttendance.html', context={}, len=0,login=session['username'])
    return redirect(url_for('Index'))

global capture
capture = 0


@app.route('/CameraAttendance', methods=['GET', 'POST'])
def CameraAttendance():
    global capture
    if 'loggedin' in session:
        if session['access']!='S':
            global capture_bool
            capture_bool = False
            if request.method == 'POST':
                global capture
                capture = 1
                print("IN capture")
                return redirect(url_for('TakeAttendance'))
            return render_template('CameraAttendance.html')
    return redirect(url_for('Index'))

def live_video():
    global capture
    cascPath = "./haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    while True:
        success, frame = camera.read()  # read the camera frame
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
                global cap_path
                cap_path=basedir+"\\capture\\"+"capture_"+str(int(time.time()))+".jpg"
                cv2.imwrite(cap_path, frame)
                global capture_bool
                capture_bool = True                
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
        user_data = pd.read_csv(maindir+"\\Notebook_Scripts_Data\\studentdetails.csv", index_col=0).T.to_dict()
        attendance_data = json.load(open(maindir+"\\Notebook_Scripts_Data\\data.json"))
        show = False
        if request.method == 'POST':
            global subject_selected_detail
            subject_selected_detail = request.form["subject"]
            show=True
            return render_template('AttendanceDetails.html',attendance_data=attendance_data, s_access = session['access'], username = session['username'] , user_data = user_data ,subject_selected=subject_selected_detail,show=show,round=round)
        else:
            return render_template('AttendanceDetails.html',attendance_data=attendance_data, s_access = session['access'], username = session['username'] , user_data = user_data,show=show)
    return redirect(url_for('Index'))

def formatter(data_required):
    attendance_data = json.load(open(maindir+"\\Notebook_Scripts_Data\\data.json"))
    if data_required[0:3]=='KCE':
        data=[['Date','Status']]
        for value in attendance_data['attendance'][subject_selected_detail]:
            if (data_required in value['absent_list']):
                temp=[value['date'],'Absent']
            else:
                temp=[value['date'],'Present']
            data.append(temp)
    else:
        data=[['Date','Status']]
        for value in attendance_data['attendance'][data_required]:
            if (session['username'] in value['absent_list']):
                temp=[value['date'],'Absent']
            else:
                temp=[value['date'],'Present']
            data.append(temp)
    return data


@app.route('/info',methods=['GET','POST'])
def info():
    if 'loggedin' in session:
        data_required=request.args.get('type')
        data=formatter(data_required)
        return render_template('info.html',data=data,zip=zip,len=len)
    return redirect(url_for('Index'))

# Running the app
if __name__ == "__main__":
    app.run(host=host_add, port=port_add, debug=True)
    app.jinja_env.filters['zip'] = zip