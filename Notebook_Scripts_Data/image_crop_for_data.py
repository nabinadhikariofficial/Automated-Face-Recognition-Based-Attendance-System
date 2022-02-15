from retinaface import RetinaFace
import cv2
import time
from os import listdir
from os.path import isfile, join
import os

basedir = os.path.abspath(os.path.dirname(__file__))

def face_detection(image_loc):
    faces = RetinaFace.detect_faces(img_path=image_loc)
    image = cv2.imread(image_loc)
    for face in faces.items():
        data = face[1]["facial_area"]
        # formula = (y1:y2+1 ,x1:x2+1)
        crop = image[data[1]:data[3]+1, data[0]:data[2]+1]
        location2 =basedir+'\\output\\' + \
            str(face[0]) + '_'+str(int(time.time())) + '.jpg'
        cv2.imwrite(location2, crop)
    return

mypath=basedir+"\\image_raw\\"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for files in onlyfiles:
    path=mypath+files
    face_detection(path)