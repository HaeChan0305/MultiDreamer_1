# temporarily made by Hanbee
from PIL import Image
image = Image.open('/root/MultiDreamer/ROCA/network/assets/sofa3.jpg')
image = image.resize((480, 480))
image = image.crop((0, 60, 480, 420)) 
image.save('/root/MultiDreamer/ROCA/network/assets/sofa3.jpg')
print(image.size)