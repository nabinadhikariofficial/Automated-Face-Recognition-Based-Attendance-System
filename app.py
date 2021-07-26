from retinaface import RetinaFace
import matplotlib.pyplot as plt
from flask import Flask, render_template


app = Flask(__name__)

face_detector = RetinaFace(quality='high')
faces = RetinaFace.detect_faces("download.jpg")
for face in faces:
    print(face)
    print('\n')


@app.route('/')
def index():
    render_template('index.html')


if __name__ == "__main__":
    app.run("0.0.0.0", 5000, True)
