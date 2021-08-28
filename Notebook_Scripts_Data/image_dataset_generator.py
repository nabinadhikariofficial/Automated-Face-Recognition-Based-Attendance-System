from PIL import Image, ImageEnhance
import os

""" CRN = "KCE074BCT005"
image = Image.open("data/im"+CRN+".jpg")

enhancer = ImageEnhance.Brightness(image)
enhancer2 = ImageEnhance.Contrast(image)

for i in range(8):
    factor = i/4.0
    im = enhancer.enhance(factor)
    im1 = enhancer2.enhance(factor)
    im.save("data/"+CRN+"_"+"B_"+str(factor)+".jpg")
    im1.save("data/"+CRN+"_"+"C_"+str(factor)+".jpg")
    im
    im1
 """
count = 0
files = os.listdir("data/KCE074BCT005")
for file in files:
    count = count+1
    print(file)
print(len(files))
print(count)
