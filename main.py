import matplotlib.pyplot as plt
import skimage.io as skio
from skimage.transform import warp
from scipy import misc
import numpy as np
import sys
import harris
import math
import random
import skimage
from skimage.draw import polygon
import skimage.io as io

POINT_COUNT = 4

def ANMS(points, eps, H):
    remain_num = 500
    radius = {}
    for center in points:
        rC, cC = center[0], center[1]
        interest_points = []
        H_C = H[rC, cC]
        for point in points:
            H_p = H[point[0], point[1]]
            if H_C < (eps * H_p):
                interest_points.append(point)
        if len(interest_points) > 0:
            radius[center] = np.amin(harris.dist2(np.array([center]), np.array(interest_points)))
    radius_values = sorted(((value, key) for (key, value) in radius.items()), reverse=True)
    top_500 = [[], []]
    for i in range(remain_num):
        top_500[0].append(radius_values[i][1][0])
        top_500[1].append(radius_values[i][1][1])

    return top_500


def tocenter(Cimg, Cpoints, otherImg, otherPoints):
    H_toC = computeHomographyMatrix(otherPoints, Cpoints)
    print(H_toC)
    H_fromC = np.linalg.inv(H_toC)
    origin_Shape = Cimg.shape
    canvas = np.zeros((origin_Shape[0]*2, origin_Shape[1]*2, 3))
    new_corners = transfer(np.array([[0, 0], [0, 3023], [4031, 3023], [4031, 0]]), H_toC).astype(np.int32)
    new_poly = polygon(np.array([new_corners[0][1], new_corners[1][1], new_corners[2][1], new_corners[3][1]]),
                       np.array([new_corners[0][0], new_corners[1][0], new_corners[2][0], new_corners[3][0]]))
    new_positions = np.array([new_poly[1], new_poly[0]]).T
    map_to_oringin = transfer(new_positions, H_fromC).astype(np.int32)
    print(np.max(map_to_oringin[:, 1]), np.max(map_to_oringin[:, 0]))
    canvas[new_poly[0],new_poly[1]] = otherImg[map_to_oringin[:, 1], map_to_oringin[:, 0]]
    return canvas


def tocenter_H(Cimg, otherImg, H_toC):
    H_fromC = np.linalg.inv(H_toC)
    origin_Shape = Cimg.shape
    canvas = np.zeros((origin_Shape[0]*2, origin_Shape[1]*2, 3))
    new_corners = transfer(np.array([[0, 0], [0, 3023], [4031, 3023], [4031, 0]]), H_toC).astype(np.int32)
    new_poly = polygon(np.array([new_corners[0][1], new_corners[1][1], new_corners[2][1], new_corners[3][1]]),
                       np.array([new_corners[0][0], new_corners[1][0], new_corners[2][0], new_corners[3][0]]))
    new_positions = np.array([new_poly[1], new_poly[0]]).T
    map_to_oringin = transfer(new_positions, H_fromC).astype(np.int32)
    print(np.max(map_to_oringin[:, 1]), np.max(map_to_oringin[:, 0]))
    canvas[new_poly[0], new_poly[1]] = otherImg[map_to_oringin[:, 1], map_to_oringin[:, 0]]
    return canvas


def find_descriptors(im, points):
    results = {}
    patch_size = 40
    subsample_size = 8
    for point in points:
        corner_left_x = point[0] - int(patch_size/2)
        corner_left_y = point[1] - int(patch_size/2)
        sample_patch = np.zeros((patch_size, patch_size))
        for i in range(patch_size):
            for j in range(patch_size):
                pixel = im[corner_left_x + i][corner_left_y + j]
                sample_patch[i][j] = pixel

        subsample_patch = skimage.transform.resize(sample_patch, (subsample_size, subsample_size))
        normalized = (subsample_patch - np.mean(subsample_patch))/np.std(subsample_patch)
        results[point] = np.reshape(normalized, (1, subsample_size**2))
    return results


def feature_match(desc_imA, desc_imB):
    threshold = .3
    results = {}
    for point_A, vector_A in desc_imA.items():
        dists = {}
        for point_B, vector_B in desc_imB.items():
            dists[point_B] = harris.dist2(vector_A, vector_B)[0][0]
        dists = sorted((value, key) for (key, value) in dists.items())

        best = dists[0]
        secondbest = dists[1]
        if best[0]/secondbest[0] < threshold:
            results[point_A] = best[1]
    return results


def RANSAC(matched_points):
    points_A = list(matched_points.keys())
    points_B = list(matched_points.values())
    results = {}
    sub_points = random.sample(range(1, len(points_A)), 4)
    subpoints_A = np.array([points_A[sub_points[0]], points_A[sub_points[1]], points_A[sub_points[2]], points_A[sub_points[3]]])
    subpoints_B = np.array([points_B[sub_points[0]], points_B[sub_points[1]], points_B[sub_points[2]], points_B[sub_points[3]]])

    H = computeHomographyMatrix(subpoints_A, subpoints_B)
    b = np.array(points_B)

    error = H @ np.hstack((points_A, np.ones((len(points_A), 1)))).T
    A = np.zeros(error.shape)
    for i in range(3):
        A[i, :] = error[i, :]/error[2, :]

    A = np.transpose(A)[:, :2]
    sqrd_err = np.sqrt((A[:, 0] - b[:, 0])**2 + (A[:, 1] - b[:, 1])**2)

    error_threshold = .5
    for i in range(len(sqrd_err)):
        if sqrd_err[i] < error_threshold:
            results[points_A[i]] = points_B[i]
    return results


def linearBlend(imA, imB, weight):
    blendedIm = imA * (1 - weight) + imB * weight
    return blendedIm


def computeHomographyMatrix(imA_points, imB_points):
    result = np.linalg.lstsq(computeAMatrix(imA_points, imB_points),
                             np.transpose(computeBVector(imB_points)))[0]
    return formatHMatrix(result)


def computeAMatrix(imA_points, imB_points):
    matrix_string = ""
    for i in range(POINT_COUNT):
        x, y = imA_points[i][0], imA_points[i][1]
        x_1, y_1 = imB_points[i][0], imB_points[i][1]
        value = "{} {} 1 0 0 0 {} {};".format(x, y, -1 * x * x_1, -1 * y * x_1) \
                + "0 0 0 {} {} 1 {} {};".format(x, y, -1 * x * y_1, -1 * y * y_1)
        matrix_string += value
    matrix_string = matrix_string[:-1]
    return np.matrix(matrix_string)


def computeBVector(imB_points):
    vector_string = ""
    for i in range(POINT_COUNT):
        x, y = imB_points[i][0], imB_points[i][1]
        vector_string += " {} {} ".format(x, y)

    return np.matrix(vector_string)


def formatHMatrix(result):
    H = np.matrix("{} {} {};".format(result[0], result[1], result[2])
                  + "{} {} {};".format(result[3], result[4], result[5])
                  + "{} {} 1".format(result[6], result[7]))
    return H


def transfer(points, H):
    shape = points.shape[0]
    ones = np.ones((shape, 1))
    Xs = np.array(points[:, 0:1])
    Ys = np.array(points[:, 1:])
    stacked = np.stack([Xs, Ys, ones], axis = 1)
    transfered = np.array(H @ stacked)
    weights = np.stack([transfered[:,2],transfered[:,2], transfered[:,2]], axis = 1)
    normalized = transfered/weights
    return normalized[:, :2]


def main():
    args = sys.argv[1:]
    imname_left = args[0]
    imname_right = args[1]
    im_left = io.imread(imname_left)
    im_left_bw = io.imread(imname_left, as_gray=True)
    im_right = io.imread(imname_right)
    im_right_bw = io.imread(imname_right, as_gray=True)
    H_left, coors_left = harris.get_harris_corners(im_left_bw)
    H_right, coors_right = harris.get_harris_corners(im_right_bw)

    points_left = []
    ANMS_points_left = []
    for i in range(len(coors_left[0])):
        points_left.append((coors_left[0][i], coors_left[1][i]))
    ANMS_coors_left = ANMS(points_left, .9, H_left)

    points_right = []
    ANMS_points_right = []
    for i in range(len(coors_right[0])):
        points_right.append((coors_right[0][i], coors_right[1][i]))
    ANMS_coors_right = ANMS(points_right, .9, H_right)

    for i in range(len(ANMS_coors_left[0])):
        ANMS_points_left.append((ANMS_coors_left[0][i], ANMS_coors_left[1][i]))
    descriptors_left = find_descriptors(im_left_bw, ANMS_points_left)

    for i in range(len(ANMS_coors_right[0])):
        ANMS_points_right.append((ANMS_coors_right[0][i], ANMS_coors_right[1][i]))
    descriptors_right = find_descriptors(im_right_bw, ANMS_points_right)

    matched_features = feature_match(descriptors_left, descriptors_right)

    RANSAC_points = {}
    for i in range(1000):
        points = RANSAC(matched_features)
        if len(points) > len(RANSAC_points):
            RANSAC_points = points

    leftpoints = list(RANSAC_points.keys())
    rightpoints = list(RANSAC_points.values())

    leftswitch = []
    for p in leftpoints:
        leftswitch.append((p[1], p[0]))

    rightswitch = []
    for p in rightpoints:
        rightswitch.append((p[1], p[0]))

    H_L = computeHomographyMatrix(leftswitch, rightswitch)
    H_R = computeHomographyMatrix(rightswitch, rightswitch)

    resultL = tocenter_H(im_right, im_left, H_L)
    resultR = tocenter_H(im_right, im_right, H_R)
    result = linearBlend(resultL, resultR, 0.5)

    io.imsave("autostitch.jpg", result)


if __name__ == "__main__":
    main()
