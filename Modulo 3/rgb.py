import os
import cv2
from PIL import Image

path = 'output_map_1/rgb/'
output_path = 'output_map_1/rgb_3/'
for file in os.listdir('output_map_1/rgb/'):
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png"):
        print("Processing image: " + file)
        im = Image.open(os.path.join(path, file))
        im_rgb = im.convert('RGB')
        im3 = im_rgb.save(output_path + im.filename.split('/')[-1])

print("Processing has finished")
