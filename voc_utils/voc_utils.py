import six
import pandas as pd
import os
from bs4 import BeautifulSoup
# from more_itertools import unique_everseen
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io


class PascalVOC (object):
    CATEGORY_NAMES = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

    PRESENCE_TRUE = 1
    PRESENCE_FALSE = -1
    PRESENCE_DIFFICULT = 0

    def __init__(self, base_dir, year='2012'):
        root_dir = os.path.join(base_dir, 'VOCdevkit', 'VOC{}'.format(year))
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.ann_dir = os.path.join(root_dir, 'Annotations')
        self.set_dir = os.path.join(root_dir, 'ImageSets', 'Main')

        train_path = os.path.join(self.set_dir, 'train.txt')
        train_image_names = pd.read_csv(train_path, delim_whitespace=True, header=None, names=['filename'])
        self.train_image_names = set(train_image_names['filename'].values)

        val_path = os.path.join(self.set_dir, 'val.txt')
        val_image_names = pd.read_csv(val_path, delim_whitespace=True, header=None, names=['filename'])
        self.val_image_names = set(val_image_names['filename'].values)

        trainval_path = os.path.join(self.set_dir, 'trainval.txt')
        trainval_image_names = pd.read_csv(trainval_path, delim_whitespace=True, header=None, names=['filename'])
        self.trainval_image_names = set(trainval_image_names['filename'].values)

    @staticmethod
    def _read_image_list(path):

        names = [line.strip() for line in open(path).readlines()]
        return [name for name in names if name != '']

    def imgs_from_category(self, category, dataset):
        """
        Summary

        Args:
            category (string): Category name as a string (from CLASS_NAMES)
            dataset (string): "train", "val", "train_val", or "test" (if available)

        Returns:
            pandas dataframe: pandas DataFrame containing a row for each image,
            the first column 'filename' gives the name of the image while the second
            column 'presence' gives one of PRESENCE_TRUE (object present in image),
            PRESENCE_FALSE (object not present in image) or PRESENCE_DIFFICULT
            (object visible but difficult recognise without substantial use of context)
        """
        if category not in self.CATEGORY_NAMES:
            raise ValueError('Unknown category {}'.format(category))
        filename = os.path.join(self.set_dir, category + "_" + dataset + ".txt")
        df = pd.read_csv(filename, delim_whitespace=True, header=None, names=['filename', 'presence'])
        return df

    def annotation_path(self, img_name):
        """
        Given an image name, get the annotation file for that image

        Args:
            img_name (string): string of the image name, relative to
                the image directory.

        Returns:
            string: file path to the annotation file
        """
        return os.path.join(self.ann_dir, img_name) + '.xml'

    def get_annotation_xml(self, img_filename):
        """
        Load annotation file for a given image.

        Args:
            img_name (string): string of the image name, relative to
                the image directory.

        Returns:
            BeautifulSoup structure: the annotation labels loaded as a
                BeautifulSoup data structure
        """
        xml = ""
        with open(self.annotation_path(img_filename)) as f:
            xml = f.readlines()
        xml = ''.join([line.strip('\t') for line in xml])
        return BeautifulSoup(xml)

    def get_objects_in_images(self, img_names):
        if isinstance(img_names, six.string_types):
            img_names = [img_names]

        data = []
        for img_name in img_names:
            anno = self.get_annotation_xml(img_name)

            fname = anno.findChild('filename').contents[0]
            objs = anno.findAll('object')
            for obj in objs:
                obj_names = obj.findChildren('name')
                cat = obj_names[0].contents[0]
                bbox = obj.findChildren('bndbox')[0]
                xmin = int(bbox.findChildren('xmin')[0].contents[0])
                ymin = int(bbox.findChildren('ymin')[0].contents[0])
                xmax = int(bbox.findChildren('xmax')[0].contents[0])
                ymax = int(bbox.findChildren('ymax')[0].contents[0])
                data.append([fname, cat, xmin, ymin, xmax, ymax])

        return pd.DataFrame(
            data, columns=['image_filename', 'category', 'xmin', 'ymin', 'xmax', 'ymax'])

    def get_objects(self, image_names=None, category=None, dataset=None):
        datasets_by_name = {'train': self.train_image_names,
                            'val': self.val_image_names,
                            'trainval': self.trainval_image_names}
        if image_names is not None:
            pass
        else:
            if dataset is None and category is None:
                image_names = self.trainval_image_names
            else:
                if category is None:
                    image_names = datasets_by_name[dataset]
                else:
                    if dataset is None:
                        dataset = 'trainval'
                    imgs = self.imgs_from_category(category, dataset)
                    imgs = imgs[imgs['presence'] == self.PRESENCE_TRUE]
                    image_names = list(imgs['filename'].values)

        objs = self.get_objects_in_images(image_names)
        if category is not None:
            objs = objs[objs['category'] == category]
        return objs


    def load_img(self, img_filename):
        """
        Load image from the filename. Default is to load in color if
        possible.

        Args:
            img_name (string): string of the image name, relative to
                the image directory.

        Returns:
            np array of float32: an image as a numpy array of float32
        """
        img_filename = os.path.join(self.img_dir, img_filename + '.jpg')
        img = skimage.img_as_float(io.imread(
            img_filename)).astype(np.float32)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img


