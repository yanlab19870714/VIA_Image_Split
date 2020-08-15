import os
import json

from skimage import io
import matplotlib.pyplot as plt

import shapely
import numpy
from rtree import index
from rtree.index import Rtree

class AnnotatedImages:
    # member variables:
    # 1. img_dir # path of the image folder
    # 2. annotation_dict # a dictionary of annotations in the format of the JSON file
    
    def __init__(self, img_dir, annotation_file_path):
        self.img_dir = img_dir
        with open(annotation_file_path) as json_file:
            self.annotation_dict = json.load(json_file)
    
    def get_path(self, img_file):
        return self.img_dir+'/'+img_file
    
    def img_size(self, img_file): # return the JSON key of VIA (= filename + file_size)
        size = os.stat(self.get_path(img_file)).st_size
        return size
    
    def img_key(self, img_file): # return the JSON key of VIA (= filename + file_size)
        size = self.img_size(img_file)
        dict_key = img_file + str(size)
        return dict_key

    def generate(self, bbox_shape, output_image_directory, output_annotation_file_path): # todo: need to add "overlap ratio"
        # todo: if output_image_directory does not exists, create the folder
        
        # slide a box of the shape: bbox_shape = (width, height)
        box_w, box_h = bbox_shape
        for img_file in os.listdir(self.img_dir):
            if img_file.endswith('.jpg'): # if current file is an .jpg image
                # 1. check size to make sure it is larger than bbox
                dict_key = self.img_key(img_file) # form the JSON key to get polygons
                print(dict_key) # for debugging
                # load the current image
                img = io.imread(self.get_path(img_file))
                img_h, img_w, _ = img.shape
                if img_h >= box_h and img_w >= box_w: # bbox can fit inside the image
                    # todo: slide x and y (maybe y goes first)
                    x = 0 # change later...
                    y = 0 # change later...
                    new_img = img[y:y+box_h, x:x+box_w]
                    io.imshow(img) # todo: instead of showing, write new_img as a file under output_image_directory
                    plt.show()
                else:
                    pass # todo: just output whatever is the input
                    # todo: maybe we need to split the else-branch if one-dimension is very long