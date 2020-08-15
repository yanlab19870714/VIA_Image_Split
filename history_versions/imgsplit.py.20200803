import os
import json

import skimage
from skimage import io
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
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
        # now output folder is hard-coded
        self.new_dir = os.path.join(os.getcwd(), r"output_image_directory")
        if not os.path.exists(self.new_dir):
            os.makedirs(self.new_dir)

    def get_path(self, img_file):
        if os.name == 'nt':
            return self.img_dir + '\\' + img_file
        return self.img_dir + '/' + img_file

    def img_size(self, img_file):  # return the JSON key of VIA (= filename + file_size)
        size = os.stat(self.get_path(img_file)).st_size
        return size

    def img_key(self, img_file):  # return the JSON key of VIA (= filename + file_size)
        size = self.img_size(img_file)
        dict_key = img_file + str(size)
        return dict_key

    def get_region_list(self, dict_key): # get the polygons (wrapped by region-object list) of an image
        regions = self.annotation_dict[dict_key]["regions"]
        return regions

    def region2polygon(self, region): # get the polygon that is wrapped in a region object
        x_and_y_unparsed = region["shape_attributes"]
        all_points_x = x_and_y_unparsed["all_points_x"]
        all_points_y = x_and_y_unparsed["all_points_y"]
        return Polygon(list(zip(all_points_x, all_points_y)))

    def get_polygon_list(self, img_file): # get the polygons (wrapped by region-object list) of an image
        dict_key = self.img_key(img_file)
        regions = self.get_region_list(dict_key)
        polygons = []
        for region in regions:
            polygon = self.region2polygon(region)
            polygons.append(polygon)
        return polygons

    # todo: need to add "overlap ratio"

    def generate(self, bbox_shape, output_image_directory, output_annotation_file_path, overlap):
        # todo: if output_image_directory does not exists, create the folder
        # slide a box of the shape: bbox_shape = (width, height)
        box_w, box_h = bbox_shape
        assert(min(box_w, box_h) > overlap)
        for img_file in os.listdir(self.img_dir):
            if img_file.endswith('.jpg'):  # if current file is an .jpg image
                # get ploygons of the img_file
                polygons = self.get_polygon_list(img_file)
                # ------> todo: build an R-tree over the polygons
                # load the current image
                img = io.imread(self.get_path(img_file), pilmode="RGB")
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!! debug BEGIN: to plot the image along with all polygons
                plt.figure(figsize=(10,10))
                io.imshow(img)
                for polygon in polygons:
                    x, y = polygon.exterior.xy
                    plt.plot(x, y)
                plt.show()
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!! debug END
                img_h, img_w, _ = img.shape
                # if bbox can fit inside the image
                if img_h >= box_h and img_w >= box_w:
                    accumulator_x, accumulator_y = 0, 0
                    for i in range(overflow(img_w, box_w, overlap)):
                        for j in range(overflow(img_h, box_h, overlap)):
                            # slide
                            y_end = accumulator_y+box_h
                            if y_end > img_h:
                                y_end = img_h
                            x_end = accumulator_x+box_w
                            if x_end > img_w:
                                x_end = img_w
                            new_img = img[accumulator_y:y_end,
                                          accumulator_x:x_end]
                            # save the image
                            img_file_noJPG = img_file[:-4]
                            new_img_path = os.path.join(self.new_dir, f"{img_file_noJPG}_{i}_{j}.jpg")
                            io.imsave(new_img_path, new_img)
                            accumulator_y += box_h - overlap
                            # ------> todo: use R-tree to get the polygons fully contained inside box [accumulator_y:y_end, accumulator_x:x_end]
                            # ------> todo: create an entry of img_i_j along with those polygons inside the output dict object to be flushed as a JSON annotation file at last
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
                break  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! debug: stop after just one iteration

an = AnnotatedImages(r"test",
                     r"test/annotations.json")
an.generate((1000, 1000), r"annotated", r"annotated/new_annotation.json", 200)

# an = AnnotatedImages(r"maskrcnn_imagecut\test",
#                     r"maskrcnn_imagecut\test\annotations.json")  # AnnotatedImages('test', 'test/annotations.json')
#an.generate((1000, 1000), r"annotated", r"annotated\new_annotation.json", 200)
