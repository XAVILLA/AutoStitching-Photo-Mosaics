import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import skimage.transform as tr
from skimage.draw import polygon
import sys
import numpy as np
import skimage.io as io
import os.path as path
import os
import time

POINT_COUNT = 4

def findPoints(img):
    print(img.shape)
    plt.imshow(img)
    i = 0
    img_points = []
    while i < POINT_COUNT:
        x = plt.ginput(1, timeout=0)
        print(x)
        img_points.append([x[0][0], x[0][1]])
        plt.scatter(x[0][0], x[0][1])
        plt.draw()
        i += 1
    plt.close()
    h, w = img.shape[:2]
    assert len(img_points) == POINT_COUNT
    return img_points


def storePoints(im):
    im_points = findPoints(im)
    f = open("datadoe2W.txt", "w")
    f.write("im_points = " + str(im_points) + "\n")
    f.close()
    return im_points



def process_data(name):
    filename = "data.txt".format(name)
    with open(filename) as file:
        data = file.readlines()[0]
    data = data[13:][:-2]
    data = data + ","
    data = data.replace(" ", "").split("[")[1:]
    data = [a.replace("],", "") for a in data]
    data = [a.split(",") for a in data]
    data = [[float(a[0]), float(a[1])] for a in data]
    print(len(data), "data points read from file")
    data = np.array(data)
    assert len(data) == POINT_COUNT + 4
    return data


image_name = "sourceImg/doe2.jpeg"
image = io.imread(image_name)
storePoints(image)