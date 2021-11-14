import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt


def print_IDs():
    print("305237257+312162027\n")


def writeMorphingVideo(image_list, video_name):
    out = cv2.VideoWriter(video_name + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, image_list[0].shape, 0)
    for im in image_list:
        out.write(im)
    out.release()


def createMorphSequence(im1, im1_pts, im2, im2_pts, t_list, transformType):
    if transformType:  # projective
        T12 = findProjectiveTransform(im1_pts, im2_pts)
        T21 = findProjectiveTransform(im2_pts, im1_pts)
    else:
        T12 = findAffineTransform(im1_pts, im2_pts)
        T21 = findAffineTransform(im2_pts, im1_pts)

    ims = []
    for t in t_list:
        T12_t = (1 - t) * np.eye(3) + (t * T12)
        T21_t = ((1 - t) * T21) + (t * np.eye(3))
        img1_t = mapImage(im1, T12_t, im1.shape)
        img2_t = mapImage(im2, T21_t, im1.shape)
        # cross-dissolve
        nim = ((1 - t) * img1_t) + (t * img2_t)
        ims.append(nim.astype(np.uint8))

    return ims


def mapImage(im, T, sizeOutIm):
    new_im = np.zeros(sizeOutIm)

    # create meshgrid of all coordinates in new image [x,y]
    xx, yy = np.meshgrid(np.arange(sizeOutIm[0]), np.arange(sizeOutIm[1]))
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    # add homogenous coord [x,y,1]
    target_coords = np.vstack((xx, yy, np.ones(xx.size))).T

    # calculate source coordinates that correspond to [x,y,1] in new image
    source_coords = np.matmul(np.linalg.inv(T), target_coords.T).T

    # todo not sure if needed
    source_coords.T[0] = source_coords.T[0] / source_coords.T[2]
    source_coords.T[1] = source_coords.T[1] / source_coords.T[2]

    # find coordinates outside range and delete (in source and target)
    out_of_range_indices = np.any((source_coords >= sizeOutIm[0] - 1) | (source_coords < 0), axis=1)
    source_coords = np.delete(source_coords, out_of_range_indices, axis=0)
    target_coords = np.delete(target_coords, out_of_range_indices, axis=0)

    source_coords = source_coords.T
    ceil_points = np.ceil(source_coords).astype(np.int)
    floor_points = np.floor(source_coords).astype(np.int)
    NE, NW, SE, SW = im[ceil_points[0], ceil_points[1]], im[floor_points[0], ceil_points[1]], im[
        ceil_points[0], floor_points[1]], im[floor_points[0], floor_points[1]]
    delta_x = source_coords[0] - floor_points[0]
    delta_y = source_coords[1] - floor_points[1]
    S = (SE * delta_x) + (SW * (1 - delta_x))
    N = (NE * delta_x) + (NW * (1 - delta_x))
    V = (N * delta_y) + (S * (1 - delta_y))

    new_im[target_coords.T[0].astype(int), target_coords.T[1].astype(int)] = V
    return new_im


def findProjectiveTransform(pointsSet1, pointsSet2):
    N = pointsSet1.shape[0]

    # iterate over points to create x , x'
    x = []
    for i in range(0, N):
        x_point = pointsSet1[i][0]
        y_point = pointsSet1[i][1]
        x_t_point = pointsSet2[i][0]
        y_t_point = pointsSet2[i][1]
        x.append([x_point, y_point, 0, 0, 1, 0, -1 * x_point * x_t_point, -1 * y_point * x_t_point])
        x.append([0, 0, x_point, y_point, 0, 1, -1 * x_point * y_t_point, -1 * y_point * y_t_point])

    x_t = pointsSet2.reshape(N * 2)
    T = np.matmul(np.linalg.pinv(x), x_t)
    T = np.array([
        [T[0], T[1], T[4]],
        [T[2], T[3], T[5]],
        [T[6], T[7], 1]
    ])

    return T


def findAffineTransform(pointsSet1, pointsSet2):
    N = pointsSet1.shape[0]

    # iterate over points to create x , x'
    x = []
    for i in range(0, N):
        x_point = pointsSet1[i][0]
        y_point = pointsSet1[i][1]
        x.append([x_point, y_point, 0, 0, 1, 0])
        x.append([0, 0, x_point, y_point, 0, 1])

    x_t = pointsSet2.reshape(N * 2)
    T = np.matmul(np.linalg.pinv(x), x_t)
    T = np.array([
        [T[0], T[1], T[4]],
        [T[2], T[3], T[5]],
        [0, 0, 1]
    ])

    return T


def getImagePts(im1, im2, varName1, varName2, nPoints):
    plt.imshow(im1, cmap='gray')
    imagePts1 = np.round(plt.ginput(n=nPoints, timeout=0))

    plt.imshow(im2, cmap='gray')
    imagePts2 = np.round(plt.ginput(n=nPoints, timeout=0))

    np.save(varName1, imagePts1)
    np.save(varName2, imagePts2)
