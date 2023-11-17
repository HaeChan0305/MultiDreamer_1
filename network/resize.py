# temporarily made by Hanbee
from PIL import Image
image = Image.open('/root/MultiDreamer/ROCA2/network/assets/bedroom_classic.jpg')
# image = image.resize((480, 480))
image = image.crop((80, 60, 560, 420)) 
image.save('/root/MultiDreamer/ROCA2/network/assets/bedroom_test.jpg')
print(image.size)