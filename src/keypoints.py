import bpy
from mathutils import Vector
from bpy import context
import numpy as np
import itertools
import cv2
from bpy_extras.object_utils import world_to_camera_view
import os
import re


class KeyPoints():
    def __init__(self, data):
        self.data = data


    def project_keypoints_to_2D_from_collection(self):
        """
        Docstring for project_keypoints_to_2D_from_collection
        
        :param self: self instance of the KeyPoints class
        :return: dictionary with collection names as keys and lists of 2D keypoint data as values

        """
        all_keypoints = {}
        for i in self.data.all_objects_collection:
            c = i.keypoints_collection
            keypoints_2d = self.project_keypoints_to_2D(c)
            all_keypoints[c.name] = keypoints_2d
        return all_keypoints

    def project_keypoints_to_2D(self, collection):
        keypoints_2d = []
        for keypoint in collection.objects:
            world_coord = keypoint.matrix_world.translation
            camera_coord = world_to_camera_view(self.data.scene, self.data.camera, world_coord)
            px = camera_coord.x * self.data.resx
            py = (1-camera_coord.y) * self.data.resy

            present_in_frame = (camera_coord.z > 0) and (0 <= camera_coord.x <= 1) and (0 <= camera_coord.y <= 1)
            #check occlusion
            camera_location = self.data.camera.matrix_world.translation
            direction = (world_coord - camera_location).normalized()
            distance = (world_coord - camera_location).length

            r, loc, n, i, obj, m = self.data.scene.ray_cast(
                depsgraph = bpy.context.evaluated_depsgraph_get(),
                origin = camera_location,
                direction = direction
            )
            visible = False
            if r:
                hit_distance = (loc - camera_location).length
                remainder = abs(distance - hit_distance)
                if remainder <= 0.006: #less than 0.6cm apart
                    visible = True
                #visible = distance < hit_distance + 1e-4

            keypoint_data = {
                "name": keypoint.name,
                "x": px,
                "y": py,
                "inFrame": present_in_frame,
                "occluded":  not visible
            }
            keypoints_2d.append(keypoint_data)
            
        keypoints_sorted = sorted(
            keypoints_2d,
            key=lambda d: d["name"].lower()
        )
        #keypoints_sorted = sorted(keypoints_2d, key=lambda kp: kp["name"]) does not work if numbers are present 
        return keypoints_sorted
