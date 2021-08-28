from PIL import Image, ImageEnhance
import os
import pandas as pd


def manipulation(image_path):
    image = Image.open(image_path)
    B_enhancer = ImageEnhance.Brightness(image)
    C_enhancer = ImageEnhance.Contrast(image)
    hoz_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
    image_loc = image_path[:-4]
    for i in range(16):
        if i > 0:
            factor = i/8.0
            im = B_enhancer.enhance(factor)
            im1 = C_enhancer.enhance(factor)
            rotate_image = image.rotate(5+i*2)
            im.save(image_loc+"_B_"+str(factor)+".jpg")
            im1.save(image_loc+"_C_"+str(factor)+".jpg")
            rotate_image.save(image_loc+"_R_"+str(5+i*5)+".jpg")
    hoz_flip.save(image_loc+"_hf.jpg")


data_crn = pd.read_csv("crnAndName.csv")
for crn in data_crn['CRN']:
    filepath = "data/"+crn+"/"
    filenames = os.listdir(filepath)
    for file in filenames:
        manipulation(filepath+file)
