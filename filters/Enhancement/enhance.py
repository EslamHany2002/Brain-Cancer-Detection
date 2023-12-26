from PIL import Image, ImageEnhance
image = Image.open('Tr-me_0080.jpg')
enhancer = ImageEnhance.Contrast(image)

factor = 2

image_output = enhancer.enhance(factor)
image_output.save("test.png")