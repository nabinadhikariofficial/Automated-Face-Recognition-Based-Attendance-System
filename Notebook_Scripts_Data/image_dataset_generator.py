from PIL import Image, ImageEnhance

CRN = "KCE074BCT005"
image = Image.open("data/"+CRN+".jpg")

B_enhancer = ImageEnhance.Brightness(image)
C_enhancer = ImageEnhance.Contrast(image)
hoz_flip = image.transpose(Image.FLIP_LEFT_RIGHT)




for i in range(16):
    if i>0:    
        factor = i/8.0
        im = B_enhancer.enhance(factor)
        im1 = C_enhancer.enhance(factor)
        rotate_image=image.rotate(5+i*2)
       
        im.save("data/"+CRN+"_"+"B_"+str(factor)+".jpg")
        im1.save("data/"+CRN+"_"+"C_"+str(factor)+".jpg")
       
        rotate_image.save("data/"+CRN+"_"+"R_"+str(5+i*5)+".jpg")
        
hoz_flip.save("data/"+CRN+"_"+"hf_"+str(factor)+".jpg")
