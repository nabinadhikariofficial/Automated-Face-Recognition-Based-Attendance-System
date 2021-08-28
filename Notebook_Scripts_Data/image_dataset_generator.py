from PIL import Image, ImageEnhance

CRN = "KCE074BCT023"
image = Image.open("data/"+CRN+".jpg")

enhancer = ImageEnhance.Brightness(image)
enhancer2 = ImageEnhance.Contrast(image)

for i in range(8):
    factor = i/4.0
    im = enhancer.enhance(factor)
    im1 = enhancer2.enhance(factor)
    im.save("data/"+CRN+"_"+"B_"+str(factor)+".jpg")
    im1.save("data/"+CRN+"_"+"C_"+str(factor)+".jpg")
