from retinaface import RetinaFace
import matplotlib.pyplot as plt

faces = RetinaFace.detect_faces("download.jpg")
for face in faces:
    print(face)
    print('\n')
