import cv2
from datetime import datetime
from retinaface import RetinaFace
vid = cv2.VideoCapture(0)
detected = 1
while True:
    ret, frame = vid.read()
    cv2.imshow("frame", frame)
    #if (cv2.waitKey(1) % 256) == 27:  # exit if escape
     #   print("Scape hit, closing....")
     #   break
    if (cv2.waitKey(1) % 256) == 32:  # snapshot if space
        date = datetime.now()
        file_name_format = "{:s}-{:04d}-{:%Y%m%d_%H%M%S}.{:s}"
        img_name = file_name_format.format("image", detected, date, "jpg")
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))

        imag_path = img_name
        faces = RetinaFace.detect_faces(imag_path)
        break
vid.release()
cv2.destroyAllWindows()
