import bpy
from mathutils import Vector
from bpy import context
import numpy as np
import itertools
import cv2
from bpy_extras.object_utils import world_to_camera_view
import os


class BoundingBox():
    def __init__(self, data):
        self.data = data

    def raycast_detect_corners_obj(self, obj):
        """
        Docstring for raycast_detect_corners_obj
        
        :param self: self instance of the BoundingBox class
        :param obj: a single mesh object in a collection
        :return: list of 2D screen coordinates of the object's vertices

        """
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()

        world_matrix = obj.matrix_world
        camera_matrix = self.data.camera.matrix_world.inverted()

        screen_coords = []

        for vertex in mesh.vertices:
            world_co = world_matrix @ vertex.co
            camera_co = camera_matrix @ world_co

            if camera_co.z < 0:
                screen_co = world_to_camera_view(self.data.scene, self.data.camera, world_co)
                px = screen_co.x * self.data.resx
                py = (1-screen_co.y) * self.data.resy
                screen_coords.append((px, py))
        return screen_coords

    #deprecate this function in the future
    #this function works only for a specific spacecraft model: a single body divided into two collections
    def raycast_detect_corners_collection(self):
        screen_coords = []
        for a in self.data.bottom_collection.all_objects:
            screen_coords.extend(self.raycast_detect_corners_obj(a))
        for b in self.data.top_collection.all_objects:
            screen_coords.extend(self.raycast_detect_corners_obj(b))

        if screen_coords:
            xs, ys = zip(*screen_coords)
            bbox = (min(xs), min(ys), max(xs), max(ys))
            return bbox
        
        return None
    
    #function to calculate bounding boxes for multi class objects
    #this function works for any number of objects divided into different collections
    def raycast_detect_corners_collection_multiclass(self):
        """
        Docstring for raycast_detect_corners_collection_multiclass
        
        :param self: self instance of the BoundingBox class
        :return: dictionary of bounding boxes for each collection

        A collection is treated as a separate class. The function iterates through all collections,
        computes the 2D screen coordinates of the vertices of all objects in each collection,
        and calculates the bounding box for each collection. The result is returned as a dictionary
        where the keys are collection names and the values are the corresponding bounding boxes.
        """
        bboxes = {}
        for i in self.data.all_objects_collection:
            c = i.collection
            screen_coords = []
            for a in c.all_objects:
                screen_coords.extend(self.raycast_detect_corners_obj(a))
            
            if screen_coords:
                xs, ys = zip(*screen_coords)
                bbox = (min(xs), min(ys), max(xs), max(ys))
            
                bboxes[c.name] = bbox
        
        return bboxes
            

    def project_bbox_to_2D_from_collection(self):
        return self.raycast_detect_corners_collection_multiclass()