import sys

import bpy
from mathutils import Vector, Euler, Quaternion, Matrix
from bpy import context
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
import random
import math
from typing import Tuple

class Bounds():
    def __init__(self, x:Tuple[float, float], y:Tuple[float, float], z:Tuple[float, float]):
        self.X = x
        self.Y = y
        self.Z = z


class RandomizerSettings():
    def __init__(self):
        self.objectBounds:Bounds = None
        self.cameraBounds:Bounds = None
        self.changeObjectPositionX:bool = True
        self.changeObjectPositionY:bool = True
        self.changeObjectPositionZ:bool = False
        self.changeCameraPositionX:bool = True
        self.changeCameraPositionY:bool = True
        self.changeCameraPositionZ:bool = True
        self.rotateObjectX:bool = False
        self.rotateObjectY:bool = False
        self.rotateObjectZ:bool = True
        self.cameraDistance:Tuple = None


class Randomizer():
    def __init__(self, data, settings = None):
        self.data = data
        if settings==None:
            self.settings = RandomizerSettings()
            self.settings.objectBounds = Bounds(x=(-2.0, 2.0), y=(-1.215, 1.215), z=(0.93, 2.0))
            self.settings.cameraBounds = Bounds(x=(-2.0, 2.0), y=(-1.215, 1.215), z=(0.93, 2.0))
        else:
            self.settings = settings

    def randomize_camera_rotation(self, max_degrees=6):
        camera = self.data.camera
        max_radians = math.radians(max_degrees)
        max_radiansz = math.radians(3)

        #camera.rotation_euler[0] += random.uniform(-max_radians, max_radians)  # X (pitch)
        camera.rotation_euler[1] += random.uniform(-max_radians, max_radians)  # Y (roll)
        camera.rotation_euler[2] += random.uniform(-max_radians, max_radians) 

    def set_minimum_distance(self):
        distance = (self.data.camera.location - self.data.obj_controller.location).length
        if self.settings.cameraDistance is not None:
            minDistance = random.uniform(*self.settings.cameraDistance)        
            d = (self.data.camera.location - self.data.obj_controller.location).normalized()
            self.data.camera.location = self.data.obj_controller.location + d * minDistance
        return distance
    
    def offset_camera_position(self, offsetVal=0.2):
        offset = offsetVal   
        ox = random.uniform(-offset, offset)
        oy= random.uniform(-offset, offset)        
        offsetVector = Vector((ox, oy, 0))
        self.data.camera.location = self.data.camera.location + offsetVector
        
    def camera_x_coverage(self, cam, distance):
        # Horizontal FOV in radians
        fov_x = 2 * math.atan(cam.data.sensor_width / (2 * cam.data.lens))
        # Full width at distance
        width = 2 * distance * math.tan(fov_x / 2)
        return width
    
    def lookAtObject(self):
        distance = (self.data.obj_controller.location - self.data.camera.location).length
        width = self.camera_x_coverage(self.data.camera, distance)
        camera_angle_offset = (-width // 3, width // 3)
        offset = random.uniform(*camera_angle_offset)    
        direction = self.data.obj_controller.location - self.data.camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        self.data.camera.rotation_euler = rot_quat.to_euler()
        #offset the angle
        theta_rad = math.atan2(offset, distance)
        local_y_world = self.data.camera.matrix_world.to_3x3() @ Vector((0, 1, 0))
        q = Quaternion(local_y_world, theta_rad)
        self.data.camera.rotation_euler = (q @ self.data.camera.rotation_euler.to_quaternion()).to_euler()
        # print(distance)
        # offset_magnitude = -6
        # local_vector = Vector((offset_magnitude, 0, 0)) # The movement we want in local space
        # world_vector = self.data.camera.matrix_world.to_3x3() @ local_vector
        # offset_location = self.data.camera.location + world_vector

        # direction = self.data.obj_controller.location - offset_location
        # rot_quat = direction.to_track_quat('-Z', 'Y')
        # self.data.camera.rotation_euler = rot_quat.to_euler()

    def randomize_camera_object_position(self):
        #bounds
        objBoundsX = self.settings.objectBounds.X
        objBoundsY = self.settings.objectBounds.Y
        objBoundsZ = self.settings.objectBounds.Z
        camBoundsX = self.settings.cameraBounds.X
        camBoundsY = self.settings.cameraBounds.Y
        camBoundsZ = self.settings.cameraBounds.Z

        rotation = (0, 360)
        reduce = 0.066
        reducedBoundsX = (objBoundsX[0]+reduce, objBoundsX[1] - reduce)
        reducedBoundsY = (objBoundsY[0]+reduce, objBoundsY[1] - reduce)
        #random object orientation and position
        obj = self.data.obj_controller
        camera = self.data.camera
        objx = obj.location.x
        objy = obj.location.y
        objz = obj.location.z
        objRx = 0
        objRy = 0
        objRz = 0
        #change coordinates
        if self.settings.changeObjectPositionX: objx = random.uniform(*reducedBoundsX)
        if self.settings.changeObjectPositionY: objy = random.uniform(*reducedBoundsY)
        if self.settings.changeObjectPositionZ: objz = random.uniform(*objBoundsZ)

        if self.settings.rotateObjectX: objRx = random.uniform(*rotation)
        if self.settings.rotateObjectY: objRy = random.uniform(*rotation)
        if self.settings.rotateObjectZ: objRz = random.uniform(*rotation)
        
        obj.location = Vector((objx, objy, objz))
        obj.rotation_euler = (math.radians(objRx), math.radians(objRy), math.radians(objRz))



        #random camera position
        if self.settings.changeCameraPositionX: camx = random.uniform(*camBoundsX)
        if self.settings.changeCameraPositionY: camy = random.uniform(*camBoundsY)
        if self.settings.changeCameraPositionZ: camz = random.uniform(*camBoundsZ)
        if random.random() > 0.05:
            self.data.camera.location = Vector((camx, camy, camz))
        distance = self.set_minimum_distance()
        self.lookAtObject()
        

        old = False
        if old:    
            offsetVal = 0.05
            if distance < 0.4:
                offsetVal = 0.015
            self.offset_camera_position(offsetVal)
            #random camera rotation
            self.randomize_camera_rotation()
            #restore actual random z position
            camera.location = Vector((camera.location.x, camera.location.y, camz))
            self.lookAtObject()

    def randomize_lights(self):
        energyR = (1, 8)
        lightProperty = self.data.lights.objects[0].data
        lightProperty.energy = random.uniform(*energyR)
        for p in self.data.lights.objects:
                p.hide_render = random.random() < 0.15





