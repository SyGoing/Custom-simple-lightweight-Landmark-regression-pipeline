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

"""
1（左上）                 2（右上）

4（左下）                 3（右下）
"""

def main():
	args = parse_args()
	resize_width= args.image_width
	resize_height=args.image_height
	data_root=args.data_root
	output_dir=args.output_dir

	images_path=os.path.join(data_root,"images")
	image_files=os.listdir(images_path)

	fw=open(os.path.join(output_dir, 'landmark_list' + '.txt'), 'w')
	fwbbox = open(os.path.join(output_dir, 'bbox_list' + '.txt'), 'w')

	crop_count=0
	for file in image_files:
		filename = os.path.splitext(file)[0]
		filetype = os.path.splitext(file)[1]
		if filetype!='.jpg':
			continue
		image_path=os.path.join(data_root,"images",file)
		json_path=os.path.join(data_root,"annotations",filename+".json")

		img = cv2.imread(image_path)
		labeldata = json.load(open(json_path))

		img_h = img.shape[0]
		img_w = img.shape[1]

		bboxes=[]
		for shape in labeldata['shapes']:
			crop_count+=1
			points = shape['points']
			pt2d_real=np.array(points)
			pt2d_x = pt2d_real[:, 0]
			pt2d_y = pt2d_real[:, 1]

			x_min = int(min(pt2d_x))
			x_max = int(max(pt2d_x))
			y_min = int(min(pt2d_y))
			y_max = int(max(pt2d_y))

			h = y_max - y_min
			w = x_max - x_min

			# ad = 0.4

			ad = random.uniform(0.3, 0.4)
			delta=ad*h


			x_min = max(int(x_min - delta), 0)
			x_max = min(int(x_max + delta), img_w - 1)
			y_min = max(int(y_min - delta), 0)
			y_max = min(int(y_max + delta), img_h - 1)

			bboxes.append([x_min,y_min,x_max,y_max])

			crop_img = img[y_min:y_max, x_min:x_max]
			w_factor = resize_width / float(crop_img.shape[1])
			h_factor = resize_height / float(crop_img.shape[0])
			crop_img = cv2.resize(crop_img, ( resize_width,resize_height))
			save_img = crop_img.copy()

			landmarks = np.array([-256.0 for i in range(4 * 2)])
			for k in range(pt2d_x.shape[0]):
				center = (int((pt2d_x[k] - x_min) * w_factor), int((pt2d_y[k] - y_min) * h_factor))
				cv2.circle(crop_img, center, 1, (255, 255, 0), 1)
				point = (center[0] / resize_width, center[1] / resize_height)
				landmarks[k * 2 + 0] =point[0]# round(point[0],6)
				landmarks[k * 2 + 1] = point[1]#round(point[1],6)


			landmark_str = ' '.join(list(map(str, landmarks.reshape(-1).tolist())))
			crop_name=filename+str(crop_count).zfill(6)+'.jpg'
			cv2.imwrite(os.path.join(output_dir,'croped_images',crop_name),save_img)
			label='croped_images/'+crop_name+" "+landmark_str+'\n'
			fw.write(label)
			if args.is_show:
				cv2.imshow('check',crop_img)
				cv2.waitKey(0)

		bboxes=np.array(bboxes)
		box_str=' '.join(list(map(str, bboxes.reshape(-1).tolist())))
		label_box=file+" "+box_str+'\n'
		fwbbox.write(label_box)

if __name__ == '__main__':
	main()








