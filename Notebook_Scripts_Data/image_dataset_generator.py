from PIL import Image, ImageEnhance

image = Image.open("data/KCE074BCT023.jpg")
print(image.size)


enhancer = ImageEnhance.Brightness(image)

for i in range(8):
    factor = i/4.0
    enhancer.enhance(factor).show("Sharpness %f", factor)
