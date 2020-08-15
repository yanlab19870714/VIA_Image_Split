import os
import json
import numpy
import matplotlib.pyplot as plt
import skimage
from skimage import io
from shapely.geometry import Polygon
import shapely
from rtree import index


def overflow(img_dimension, box_dimension, overlap):
    # Returns number of passes box must make in one direction to cover the image
    adjusted = box_dimension - overlap
    return (img_dimension//adjusted + int(bool(img_dimension % adjusted)))


def region2polygon(region):  # get the polygon that is wrapped in a region object
    x_and_y_unparsed = region["shape_attributes"]
    all_points_x = x_and_y_unparsed["all_points_x"]
    all_points_y = x_and_y_unparsed["all_points_y"]
    metadata = region["region_attributes"]["Class"]
    return [Polygon(list(zip(all_points_x, all_points_y))), metadata]

# build an rtree over all ploygons (their bounding boxes) in an image


def rtree_over_pgons(polygons):  # [x1, y1, x2, y2]
    idx = index.Index()
    for pos, polygon in enumerate(polygons):  # polygon[1] is its class
        idx.insert(pos, polygon[0].bounds)
    return idx


def rect2polygon(x1, y1, x2, y2):
    p1 = (x1, y1)
    p2 = (x1, y2)
    p3 = (x2, y2)
    p4 = (x2, y1)
    return Polygon([p1, p2, p3, p4])


def coord_adjust(polygon_list, positions, x1, y1):
    adjusted_polygons = []
    for pos in positions:
        polygon = polygon_list[pos]
        polygon[0] = shapely.affinity.translate(polygon[0], xoff=-x1, yoff=-y1)
        adjusted_polygons.append(polygon)
    return adjusted_polygons


def region_maker(polygons):
    regions_dicts = []
    for i in polygons:
        # https://stackoverflow.com/a/47519098/9295513
        xy_combined = list(zip(*i[0].exterior.coords.xy))
        xs, ys = zip(*xy_combined)
        # Convert the tuples to lists and the floats to ints
        xs = [int(j) for j in xs]
        ys = [int(j) for j in ys]
        regions_dicts.append({
            "shape_attributes": {
                "name": "polygon",
                "all_points_x": xs,
                "all_points_y": ys
            },
            "region_attributes": {
                "Class": i[1]
            }
        })

    return regions_dicts


class AnnotatedImages:
    # member variables:
    # 1. img_dir # path of the image folder
    # 2. annotation_dict # a dictionary of annotations in the format of the JSON file

    def __init__(self, img_dir, annotation_file_path, output_annotation_file_path, output_image_directory):
        self.img_dir = img_dir
        with open(annotation_file_path) as json_file:
            self.annotation_dict = json.load(json_file)
        # now output folder is hard-coded
        self.new_dir = os.path.join(os.getcwd(), output_image_directory)
        if not os.path.exists(self.new_dir):
            os.makedirs(self.new_dir)
        self.output_annotation_file_path = output_annotation_file_path

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

    # get the polygons (wrapped by region-object list) of an image
    def get_region_list(self, dict_key):
        regions = self.annotation_dict[dict_key]["regions"]
        return regions

    # get the polygons (wrapped by region-object list) of an image
    def get_polygon_list(self, img_file):
        dict_key = self.img_key(img_file)
        regions = self.get_region_list(dict_key)
        polygons = []
        for region in regions:
            polygon = region2polygon(region)
            polygons.append(polygon)
        return polygons

    def generate(self, bbox_shape, overlap):
        # slide a box of the shape: bbox_shape = (width, height)
        box_w, box_h = bbox_shape
        assert(min(box_w, box_h) > overlap)
        annotation_dict = {}
        for img_file in os.listdir(self.img_dir):
            if img_file.endswith('.jpg'):  # if current file is an .jpg image
                # get ploygons of the img_file
                polygons = self.get_polygon_list(img_file)
                pgon_rtree = rtree_over_pgons(polygons)
                # load the current image
                img = io.imread(self.get_path(img_file), pilmode="RGB")
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! debug BEGIN: to plot the image along with all polygons
                """
                plt.figure(figsize=(10, 10))
                io.imshow(img)
                for polygon in polygons:
                    x, y = polygon[0].exterior.xy
                    plt.plot(x, y)
                plt.show()
                # """
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! debug END
                img_h, img_w, _ = img.shape
                # if bbox can fit inside the image
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
                        # get ploygons inside the sliding box
                        query_box = (accumulator_x, accumulator_y,
                                     x_end, y_end)  # [x1, y1, x2, y2]
                        pgon_positions = list(
                            pgon_rtree.intersection(query_box))
                        # get the polygon object for the image, to be used for rtree refinement
                        sliding_box = rect2polygon(*query_box)
                        # refinement: filtering polygons that are not totally inside the sliding box
                        refined_positions = []
                        for pos in pgon_positions:
                            polygon = polygons[pos]
                            if sliding_box.contains(polygon[0]):
                                refined_positions.append(pos)

                        # print('len(refined_positions) =', len(refined_positions))  # !!!!!!!!!!!!!!!!!!!!!

                        adjusted_polygons = coord_adjust(
                            polygons, refined_positions, accumulator_x, accumulator_y)

                        if len(adjusted_polygons) == 0:  # there's no polygon in the box
                            continue

                        # get the image
                        new_img = img[accumulator_y:y_end,
                                      accumulator_x:x_end]
                        # save the image
                        img_file_noJPG = img_file[:-4]
                        imgcut_name = f"{img_file_noJPG}_{i}_{j}.jpg"
                        new_img_path = os.path.join(
                            self.new_dir, imgcut_name)
                        io.imsave(new_img_path, new_img)

                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! debug BEGIN: to plot the image along with all polygons
                        '''
                        plt.figure(figsize=(10, 10))
                        io.imshow(new_img)

                        # polyons inside
                        for polygon in adjusted_polygons:
                            x, y = polygon[0].exterior.xy
                            plt.plot(x, y)
                        plt.show()
                        # '''
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! debug END
                        # return  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! debug: stop after just one slide

                        # self.img_size(new_img_path) # this is wrong
                        imgcut_size = 241543903
                        annotation_dict[imgcut_name + str(imgcut_size)] = {
                            "filename": imgcut_name,
                            "size": imgcut_size,
                            "regions": region_maker(adjusted_polygons),
                            "file_attributes": {}
                        }

                        accumulator_y += box_h - overlap
                    # move y to beginning, move x to next row
                    accumulator_x += box_w - overlap
                    accumulator_y = 0

                # return  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! debug: stop after just one slide

        # print(annotation_dict)

        with open(self.output_annotation_file_path, 'w') as f:
            json.dump(annotation_dict, f)


"""
an = AnnotatedImages(r"test",
                     r"test/annotations.json")
an.generate((1000, 1000), r"annotated", r"annotated/new_annotation.json", 200)

"""
an = AnnotatedImages(r"test", r"test\annotations.json",
                     r"annotated\new_annotation.json", r"annotated")
an.generate((1000, 1000), 200)
# """
