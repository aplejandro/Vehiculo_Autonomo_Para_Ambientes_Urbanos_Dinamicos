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

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    #print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())

    ego_vehicle = world.spawn_actor(bp, spawn_point)
    ego_vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))
    # vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.

    actor_list.append(ego_vehicle)

    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    rgb_bp = blueprint_library.find('sensor.camera.rgb')
    depth_bp = blueprint_library.find('sensor.camera.depth')
    sem_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    # change the dimensions of the image
    rgb_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
    rgb_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
    rgb_bp.set_attribute('fov', f'{FOV}')
    depth_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
    depth_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
    depth_bp.set_attribute('fov', f'{FOV}')
    sem_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
    sem_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
    sem_bp.set_attribute('fov', f'{FOV}')

    # Adjust sensor relative to vehicle
    cam_location = carla.Location(2.25,0,2)
    cam_rotation = carla.Rotation(0,0,0)
    cam_transform = carla.Transform(cam_location,cam_rotation)

    # spawn the sensor and attach to vehicle.
    rgb_cam = world.spawn_actor(rgb_bp,cam_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
    depth_cam = world.spawn_actor(depth_bp,cam_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
    sem_cam = world.spawn_actor(sem_bp,cam_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)

    # add sensor to list of actors
    actor_list.append(rgb_cam)
    actor_list.append(depth_cam)
    actor_list.append(sem_cam)

    # do something with this sensor
    #rgb_cam.listen(lambda data: process_img(data))
    #rgb_cam.listen(lambda image: image.save_to_disk('output/rgb/%.6d.jpg' % image.frame))
    #depth_cam.listen(lambda image: image.save_to_disk('output/depth/%.6d.jpg' % image.frame,carla.ColorConverter.LogarithmicDepth))
    #sem_cam.listen(lambda image: image.save_to_disk('output/semantic_segmentation/%.6d.jpg' % image.frame,carla.ColorConverter.CityScapesPalette))

    time.sleep(20)

finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')
