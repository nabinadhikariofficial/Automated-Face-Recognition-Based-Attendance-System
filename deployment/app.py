from retinaface import RetinaFace
from matplotlib import pyplot as plt
from flask import Flask, request, render_template, jsonify, Markup, session, redirect, url_for,Response
import os
from werkzeug.utils import secure_filename
import cv2
import time

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

basedir = os.path.abspath(os.path.dirname(__file__))

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


    


@app.route('/takeattendance' , methods=['GET', 'POST'])
def takeattendance():
    return render_template('takeattendance.html')

camera = cv2.VideoCapture(0)
def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/capture')
def capture():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendancedetails' , methods=['GET', 'POST'])
def attendancedetails():
    return render_template('attendancedetails.html')
# Running the app
if __name__ == "__main__":
    app.run(host=host_add, port=port_add, debug=True)
    app.jinja_env.filters['zip'] = zip
