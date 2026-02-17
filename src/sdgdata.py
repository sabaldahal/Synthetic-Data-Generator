import bpy

class SDGData():
    def __init__(self, scene, camera, resx, resy, all_objects_collection, lights):
        self.scene = scene
        self.camera = camera
        self.resx = resx
        self.resy = resy
        self.lights = lights
        self.all_objects_collection = all_objects_collection

class ObjectInfo:
    def __init__(self, collection_name, controller_name, keypoints_collection_name = None):
        self.collection_name = collection_name
        self.controller_name = controller_name
        self.keypoints_collection_name = keypoints_collection_name
        self.controller = self._getController()
        self.collection = self._getCollection()
        self.keypoints_collection = self._getKeypointsCollection()

    def _getController(self):
        bpy.data.objects.get(self.controller_name)

    def _getCollection(self):
        bpy.data.objects.get(self.collection_name)

    def _getKeypointsCollection(self):
        bpy.data.objects.get(self.keypoints_collection_name)

class Settings:
    detectkeypoints = False
    multiclass_singlebody = False
    resize_images = False
    export_coco = True
    export_yolo = False
    resize_x = 640
    resize_y = 640