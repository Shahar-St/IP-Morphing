import os
from hw2_functions import *


def main():
    img_1_name = 'Face1'
    img_2_name = 'Face2'
    img_suffix = '.tif'
    images_dir = 'FaceImages'
    path_to_image1 = os.path.join(images_dir, img_1_name + img_suffix)
    path_to_image2 = os.path.join(images_dir, img_2_name + img_suffix)
    img1 = cv2.imread(path_to_image1)
    img2 = cv2.imread(path_to_image2)

    print("Start running script  ------------------------------------\n")
    print_IDs()

    print("a ------------------------------------\n")
    points_dir = 'FacePoints'
    if not os.path.exists(points_dir):
        os.makedirs(points_dir)

    path_to_points1 = os.path.join(points_dir, img_1_name + '_Points')
    path_to_points2 = os.path.join(points_dir, img_2_name + '_Points')
    getImagePts(img1, img2, path_to_points1, path_to_points2, nPoints=1)


if __name__ == '__main__':
    main()
