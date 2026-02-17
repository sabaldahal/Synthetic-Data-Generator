
import json
import os
import numpy as np
import bpy
from mathutils import Matrix

class TransformationMatrix():
    def __init__(self, data):
        self.data = data

    def calculateMatrix(self):                
        obj = self.data.obj_controller
        cam = self.data.camera

        obj_world = obj.matrix_world
        cam_world = cam.matrix_world
        cam_mat_world_inv = cam_world.inverted()

        # extract translation + rotation only (no scale)
        cam_rot = cam_world.to_quaternion().to_matrix().to_4x4()   # pure rotation
        cam_trans = Matrix.Translation(cam_world.to_translation()) # pure translation
        # rebuild camera world transform (rotation + translation, no scale)
        cam_world_rigid = cam_trans @ cam_rot
        # invert and apply to object
        obj_in_cam = cam_world_rigid.inverted() @ obj_world

        # convert to numpy
        M = np.array(obj_in_cam)

        return M

    