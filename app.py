from retinaface import RetinaFace
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify, Markup, session, redirect, url_for

app = Flask(__name__)
app.secret_key = 'key'
""" faces = RetinaFace.detect_faces("download.jpg")
for face in faces:
    print(face) """

host_add = '127.0.0.1'
port_add = 5000


@app.route('/')
def index():
    return render_template('login.html')


# Running the app
if __name__ == "__main__":
    app.run(host=host_add, port=port_add, debug=True)
