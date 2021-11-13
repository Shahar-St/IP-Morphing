import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt

def print_IDs():
    print("305237257+312162027\n")

def writeMorphingVideo(image_list, video_name):
    out = cv2.VideoWriter(video_name+".mp4", cv2.VideoWriter_fourcc(*'MP4V'), 20.0, image_list[0].shape, 0)
    for im in image_list:
        out.write(im)
    out.release()


# def createMorphSequence (im1, im1_pts, im2, im2_pts, t_list, transformType):
#     if transformType:
#         # TODO: projective transforms
#     else:
#         # TODO: affine transforms
#     ims = []
#     for t in t_list:
#         # TODO: calculate nim for each t
#         ims.append(nim)
#     return ims
#
#
# def mapImage(im, T, sizeOutIm):
#
#     im_new = np.zeros(sizeOutIm)
#     # create meshgrid of all coordinates in new image [x,y]
#
#
#     # add homogenous coord [x,y,1]
#
#
#     # calculate source coordinates that correspond to [x,y,1] in new image
#
#
#     # find coordinates outside range and delete (in source and target)
#
#
#     # interpolate - bilinear
#
#
#     # apply corresponding coordinates
#     # new_im [ target coordinates ] = old_im [ source coordinates ]
#
#
#
# def findProjectiveTransform(pointsSet1, pointsSet2):
#     N = pointsSet1.shape[0]
#
#     # iterate iver points to create x , x'
#     for i in range(0, N):
#
#
#     # calculate T - be careful of order when reshaping it
#     return T
#
#
def findAffineTransform(pointsSet1, pointsSet2):
    N = pointsSet1.shape[0]

    # iterate over points to create x , x'
    first_row = [0, 0, 0, 0, 1, 0]
    sec_row = [0, 0, 0, 0, 0, 1]
    x = np.array([
        first_row, sec_row,
        first_row, sec_row,
        first_row, sec_row
    ])
    points_location = 0
    for i in range(0, 3):
        x[points_location][0] = x[points_location + 1][2] = pointsSet1[i][0]
        x[points_location][1] = x[points_location + 1][3] = pointsSet1[i][1]
        points_location += 2

    x_t = pointsSet2.reshape(N * 2)[:6]
    T = np.matmul(np.linalg.pinv(x), x_t)

    return T


def getImagePts(im1, im2, varName1, varName2, nPoints):

    plt.imshow(im1)
    imagePts1 = np.asarray(plt.ginput(n=nPoints, timeout=0))

    plt.imshow(im2)
    imagePts2 = np.asarray(plt.ginput(n=nPoints, timeout=0))
    print(imagePts1)
    print(imagePts2)

    np.save(varName1, imagePts1)
    np.save(varName2, imagePts2)
