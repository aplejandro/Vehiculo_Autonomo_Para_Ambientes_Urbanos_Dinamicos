import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
import time
import numpy as np
import cv2

IM_WIDTH = 608
IM_HEIGHT = 608
FOV = 105


def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    cv2.waitKey(1)
    #return i3/255.0


actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(4.0)

    world = client.get_world()

    world=client.load_world('Town01')



    time.sleep(10)

finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')
