import os
import json

import skimage
from skimage import io
import matplotlib.pyplot as plt

import shapely
import numpy
from rtree import index
from rtree.index import Rtree

def overflow(img_dimension, box_dimension, overlap):
    # Returns number of passes box must make in one direction to cover the image
    adjusted = box_dimension - overlap
    return (img_dimension//adjusted + int(bool(img_dimension % adjusted)))

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

    def img_size(self, img_file):  # return the JSON key of VIA (= filename + file_size)
        size = os.stat(self.get_path(img_file)).st_size
        return size

    def img_key(self, img_file):  # return the JSON key of VIA (= filename + file_size)
        size = self.img_size(img_file)
        dict_key = img_file + str(size)
        return dict_key

    # todo: need to add "overlap ratio"
    def generate(self, bbox_shape, output_image_directory, output_annotation_file_path, overlap):
        # todo: if output_image_directory does not exists, create the folder
        # slide a box of the shape: bbox_shape = (width, height)
        box_w, box_h = bbox_shape
        assert(min(box_w, box_h) > overlap)
        for img_file in os.listdir(self.img_dir):
            if img_file.endswith('.jpg'):  # if current file is an .jpg image
                # 1. check size to make sure it is larger than bbox
                # form the JSON key to get polygons
                dict_key = self.img_key(img_file)
                print(dict_key)  # for debugging
                # load the current image
                img = io.imread(self.get_path(img_file))
                io.imshow(img); plt.show() #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                img_h, img_w, _ = img.shape
                if img_h >= box_h and img_w >= box_w:  # bbox can fit inside the image
                    accumulator_x, accumulator_y = 0, 0

                    print(overflow(img_w, box_w, overlap))
                    print(overflow(img_h, box_h, overlap))
                    print(f"{img_h}, {img_w}")

                    for i in range(overflow(img_w, box_w, overlap)):
                        for j in range(overflow(img_h, box_h, overlap)):
                            # todo: slide x and y (maybe y goes first)

                            print()
                            print(accumulator_x)
                            print(accumulator_y)
                            print()

                            y_end = accumulator_y+box_h
                            if y_end > img_h:
                                y_end = img_h
                            x_end = accumulator_x+box_w
                            if x_end > img_w:
                                x_end = img_w
                            new_img = img[accumulator_y:y_end,
                                          accumulator_x:x_end]
                            # todo: instead of showing, write new_img as a file under output_image_directory

                            io.imshow(new_img)
                            plt.show()
                            
                            accumulator_y += box_h - overlap
                        # move y to beginning, move x to next row
                        accumulator_x += box_w - overlap
                        accumulator_y = 0
                elif img_h <= box_h and img_w >= box_w:  # placeholder for too tall
                    pass
                elif img_h >= box_h and img_w <= box_w:  # placeholder for too wide
                    pass
                else:
                    pass
                    # todo: just output whatever is the input
                    # todo: maybe we need to split the else-branch if one-dimension is very long
                break


an = AnnotatedImages(r"test",
                     r"test/annotations.json")
an.generate((1000, 1000), r"annotated", r"annotated/new_annotation.json", 200)
