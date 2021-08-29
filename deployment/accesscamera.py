# import the opencv library
import cv2
import sys   
import logging as log
import datetime as dt
from time import sleep


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

# define a video capture object
video_capture = cv2.VideoCapture(0)

  
while(True):
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass


    ret, image = video_capture.read()




    # Display the resulting image
    cv2.imshow('Video', image)

    if cv2.waitKey(1) & 0xFF == ord('s'): 

        check, image = video_capture.read()
        cv2.imshow("Capturing", image)
        cv2.imwrite(filename='saved_img.jpg', img=image)
        #video_capture.release()
        #img_new = cv2.imread('saved_img.jpg')
        #img_new = cv2.imshow("Captured Image", img_new)
        #cv2.waitKey(1650)
        #cv2.destroyAllWindows()

        #break
    elif cv2.waitKey(1) & 0xFF == ord('a'):
    
        video_capture.release()
        cv2.destroyAllWindows()
        break


    


video_capture.release()
cv2.destroyAllWindows()