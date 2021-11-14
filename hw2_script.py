import os

from hw2_functions import *


def main():
    img_1_name = 'Face1'
    img_2_name = 'Face6'
    img_suffix = '.tif'
    images_dir = 'FaceImages'
    path_to_image1 = os.path.join(images_dir, img_1_name + img_suffix)
    path_to_image2 = os.path.join(images_dir, img_2_name + img_suffix)
    img1 = cv2.cvtColor(cv2.imread(path_to_image1), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(cv2.imread(path_to_image2), cv2.COLOR_BGR2GRAY)

    print("Start running script  ------------------------------------\n")
    print_IDs()

    print("a ------------------------------------\n")
    points_dir = 'FacePoints'
    if not os.path.exists(points_dir):
        os.makedirs(points_dir)

    path_to_points1 = os.path.join(points_dir, img_1_name + '_Points.npy')
    path_to_points2 = os.path.join(points_dir, img_2_name + '_Points.npy')
    # getImagePts(img1, img2, path_to_points1, path_to_points2, nPoints=12)

    point_set1 = np.load(path_to_points1).astype(int)
    point_set2 = np.load(path_to_points2).astype(int)

    nu_of_frames = 100
    seq = createMorphSequence(img1, point_set1, img2, point_set2, np.linspace(0, 1, nu_of_frames), 0)
    writeMorphingVideo(seq, 'outputVideo')


if __name__ == '__main__':
    main()
