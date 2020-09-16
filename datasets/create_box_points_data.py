#!encoding=utf-8
import cv2
import os
import numpy as np
import argparse
import random
import json


def parse_args():
    parser = argparse.ArgumentParser(description="This scriptand creates database for training. ",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", type=str, default='../data/original_data',
                        help="path to database")
    parser.add_argument("--output_dir", type=str, default='../data/processed_data',
                        help="path to output file")
    parser.add_argument("--image_width", type=int, default=120,
                        help="output image size")
    parser.add_argument("--image_height", type=int, default=48,
                        help="output image size")
    parser.add_argument("--is_show", type=bool, default=True,
                        help="output image size")
    args = parser.parse_args()
    return args


classnams={"plate"}

def main():
    args = parse_args()
    resize_width = args.image_width
    resize_height = args.image_height
    data_root = args.data_root
    output_dir = args.output_dir

    images_path = os.path.join(data_root, "images")
    image_files = os.listdir(images_path)

    fw = open(os.path.join(output_dir, 'landmark_list_test' + '.txt'), 'w')

    crop_count = 0
    for file in image_files:
        filename = os.path.splitext(file)[0]
        filetype = os.path.splitext(file)[1]

        image_path = os.path.join(data_root, "images", file)
        json_path = os.path.join(data_root, "annotations", filename + ".json")
        img = cv2.imread(image_path)

        if filetype != ".jpg":
            continue
        if not os.path.exists(json_path):
            continue

        labeldata = json.load(open(json_path))
        shapes = labeldata['shapes']
        num = int(len(shapes) / 2)

        for id in range(num):
            bbox = shapes[2 * id ]
            points_b=bbox['points']
            x_min = points_b[0][0]
            y_min = points_b[0][1]
            x_max = points_b[1][0]
            y_max = points_b[1][1]

            polygon = shapes[2 * id+1]
            polygon_points=polygon['points']
            pt2d_real = np.array(polygon_points)
            pt2d_x = pt2d_real[:, 0]
            pt2d_y = pt2d_real[:, 1]

            crop_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
            w_factor = resize_width / float(x_max-x_min)
            h_factor = resize_height / float(y_max-y_min)
            crop_img = cv2.resize(crop_img, (resize_width, resize_height))
            save_img = crop_img.copy()

            landmarks = np.array([-256.0 for i in range(4 * 2)])
            for k in range(pt2d_x.shape[0]):
                center = (int((pt2d_x[k] - x_min) * w_factor), int((pt2d_y[k] - y_min) * h_factor))
                cv2.circle(crop_img, center, 1, (255, 255, 0), 1)
                point = (center[0] / resize_width, center[1] / resize_height)
                landmarks[k * 2 + 0] =point[0]
                landmarks[k * 2 + 1] = point[1]
            landmark_str = ' '.join(list(map(str, landmarks.reshape(-1).tolist())))
            crop_name = filename + str(crop_count).zfill(6) + '.jpg'
            cv2.imwrite(os.path.join(output_dir, 'croped_images', crop_name), save_img)
            label = 'croped_images/' + crop_name + " " + landmark_str + '\n'
            fw.write(label)
            if args.is_show:
                cv2.imshow('check', crop_img)
                cv2.waitKey(0)


if __name__ == '__main__':
    main()








