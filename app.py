from retinaface import RetinaFace
import matplotlib.pyplot as plt
from flask import Flask, render_template

app = Flask(__name__)
faces = RetinaFace.detect_faces("download.jpg")
for face in faces:
    print(face)


@app.route('/')
def index():
    render_template('login.html')


if __name__ == "__main__":
    app.run("0.0.0.0", 5000, True)
