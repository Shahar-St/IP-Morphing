import os

from hw2_functions import *


def main():
    img_1_name = 'Face4'
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

    points_dir_section_A = 'FacePoints_section_A'
    if not os.path.exists(points_dir_section_A):
        os.makedirs(points_dir_section_A)

    path_to_points1 = os.path.join(points_dir_section_A, img_1_name + '_Points.npy')
    path_to_points2 = os.path.join(points_dir_section_A, img_2_name + '_Points.npy')
    # getImagePts(img1, img2, path_to_points1, path_to_points2, nPoints=12)

    point_set1 = np.load(path_to_points1)
    point_set2 = np.load(path_to_points2)

    num_of_frames = 100
    seq = createMorphSequence(img1, point_set1, img2, point_set2, np.linspace(0, 1, num_of_frames), 1)
    writeMorphingVideo(seq, 'outputVideo_section_A')


    print("b ------------------------------------\n")

    img_section_B_name_rectangle = 'Rectangle'
    img_section_B_name_trapeze = 'Trapeze'
    dir_section_B = 'Section_B'
    if not os.path.exists(dir_section_B):
        os.makedirs(dir_section_B)

    path_to_img_section_B_rectangle = os.path.join(dir_section_B, img_section_B_name_rectangle + img_suffix)
    img_section_B_rectangle = cv2.cvtColor(cv2.imread(path_to_img_section_B_rectangle), cv2.COLOR_BGR2GRAY)
    path_to_img_section_B_trapeze = os.path.join(dir_section_B, img_section_B_name_trapeze + img_suffix)
    img_section_B_trapeze = cv2.cvtColor(cv2.imread(path_to_img_section_B_trapeze), cv2.COLOR_BGR2GRAY)

    path_to_set1_section_B = os.path.join(dir_section_B, img_section_B_name_rectangle + '_set' + '_Points.npy')
    path_to_set2_section_B = os.path.join(dir_section_B, img_section_B_name_trapeze + '_set' + '_Points.npy')
    # getImagePts(img_section_B_rectangle, img_section_B_trapeze, path_to_set1_section_B, path_to_set2_section_B, 4)

    point_section_B_set_rectangle = np.load(path_to_set1_section_B)
    point_section_B_set_trapeze = np.load(path_to_set2_section_B)

    T_affine = findAffineTransform(point_section_B_set_rectangle, point_section_B_set_trapeze)
    T_projective = findProjectiveTransform(point_section_B_set_rectangle, point_section_B_set_trapeze)
    img_by_affine = mapImage(img_section_B_rectangle, T_affine, img_section_B_rectangle.shape)
    img_by_projective = mapImage(img_section_B_rectangle, T_projective, img_section_B_rectangle.shape)

    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(img_section_B_rectangle, cmap='gray', vmin=0, vmax=255)
    plt.title("source image")
    plt.subplot(1, 4, 2)
    plt.imshow(img_section_B_trapeze, cmap='gray', vmin=0, vmax=255)
    plt.title("target image")
    plt.subplot(1, 4, 3)
    plt.imshow(img_by_affine, cmap='gray', vmin=0, vmax=255)
    plt.title("by affine")
    plt.subplot(1, 4, 4)
    plt.imshow(img_by_projective, cmap='gray', vmin=0, vmax=255)
    plt.title("by projective")
    plt.show()


    print("c - difference between quantity of points ------------------------------------\n")

    points_dir_section_C = 'FacePoints_section_C'
    if not os.path.exists(points_dir_section_C):
        os.makedirs(points_dir_section_C)

    path_to_points1 = os.path.join(points_dir_section_C, img_1_name + '_12_Points.npy')
    path_to_points2 = os.path.join(points_dir_section_C, img_2_name + '_12_Points.npy')
    # getImagePts(img1, img2, path_to_points1, path_to_points2, nPoints=12)

    point_12_set1 = np.load(path_to_points1)
    point_12_set2 = np.load(path_to_points2)

    seq_12 = createMorphSequence(img1, point_12_set1, img2, point_12_set2, np.linspace(0, 1, num_of_frames), 1)

    path_to_points1 = os.path.join(points_dir_section_C, img_1_name + '_4_Points.npy')
    path_to_points2 = os.path.join(points_dir_section_C, img_2_name + '_4_Points.npy')
    # getImagePts(img1, img2, path_to_points1, path_to_points2, nPoints=4)

    point_4_set1 = np.load(path_to_points1)
    point_4_set2 = np.load(path_to_points2)

    seq_4 = createMorphSequence(img1, point_4_set1, img2, point_4_set2, np.linspace(0, 1, num_of_frames), 1)

    im_12 = seq_12[int(num_of_frames/2)]
    im_4 = seq_4[int(num_of_frames/2)]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im_12, cmap='gray', vmin=0, vmax=255)
    plt.title("im with 12 points")
    plt.subplot(1, 2, 2)
    plt.imshow(im_4, cmap='gray', vmin=0, vmax=255)
    plt.title("im with 4 points")
    plt.show()


    print("c - difference between distribution of the points ------------------------------------\n")

    path_to_points1 = os.path.join(points_dir_section_C, img_1_name + '_well_distribution_Points.npy')
    path_to_points2 = os.path.join(points_dir_section_C, img_2_name + '_well_distribution_Points.npy')
    # getImagePts(img1, img2, path_to_points1, path_to_points2, nPoints=12)

    point_well_distribution_set1 = np.load(path_to_points1)
    point_well_distribution_set2 = np.load(path_to_points2)

    seq_well_distribution = createMorphSequence(img1, point_well_distribution_set1, img2,
                                                point_well_distribution_set2, np.linspace(0, 1, num_of_frames), 1)


    path_to_points1 = os.path.join(points_dir_section_C, img_1_name + '_small_distribution_Points.npy')
    path_to_points2 = os.path.join(points_dir_section_C, img_2_name + '_small_distribution_Points.npy')
    # getImagePts(img1, img2, path_to_points1, path_to_points2, nPoints=12)

    point_small_distribution_set1 = np.load(path_to_points1)
    point_small_distribution_set2 = np.load(path_to_points2)

    seq_small_distribution = createMorphSequence(img1, point_small_distribution_set1, img2,
                                                 point_small_distribution_set2, np.linspace(0, 1, num_of_frames), 1)

    im_well_distribution = seq_well_distribution[int(num_of_frames/2)]
    im_small_distribution = seq_small_distribution[int(num_of_frames/2)]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im_well_distribution, cmap='gray', vmin=0, vmax=255)
    plt.title("im with well distribution")
    plt.subplot(1, 2, 2)
    plt.imshow(im_small_distribution, cmap='gray', vmin=0, vmax=255)
    plt.title("im with small distribution")
    plt.show()


if __name__ == '__main__':
    main()
