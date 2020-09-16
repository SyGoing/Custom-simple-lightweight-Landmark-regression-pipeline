#-*- coding:utf-8-*-

import xml.dom
import xml.dom.minidom
import os
import cv2
import json
import numpy as np
import argparse

_INDENT = '' * 4
_NEW_LINE = '\n'
_FOLDER_NODE = 'MULTITASK'
_ROOT_NODE = 'annotation'
_DATABASE_NAME = 'LOGODetection'
_ANNOTATION = 'LIUPANSHU'
_AUTHOR = 'SyGoing'
_SEGMENTED = '0'
_DIFFICULT = '0'
_TRUNCATED = '0'
_POSE = 'Unspecified'

_ANNOTATION_SAVE_PATH = 'E:/trainLiuPanShuiXML'

# 封装创建节点的过程
def createElementNode(doc, tag, attr):  # 创建一个元素节点
    element_node = doc.createElement(tag)

    # 创建一个文本节点
    text_node = doc.createTextNode(attr)

    # 将文本节点作为元素节点的子节点
    element_node.appendChild(text_node)

    return element_node

# 封装添加一个子节点
def createChildNode(doc, tag, attr, parent_node):
    child_node = createElementNode(doc, tag, attr)

    parent_node.appendChild(child_node)

# object节点比较特殊
def createObjectNode(doc, bbox,keypoints,classname):
    object_node = doc.createElement('object')
    midname=classname
    createChildNode(doc, 'name', midname,
                    object_node)

    createChildNode(doc, 'pose',
                    _POSE, object_node)

    createChildNode(doc, 'truncated',
                    _TRUNCATED, object_node)

    createChildNode(doc, 'difficult',
                    _DIFFICULT, object_node)

    ###------boundingbox
    xmin=bbox[0]
    ymin=bbox[1]
    xmax=bbox[2]
    ymax=bbox[3]
    bndbox_node = doc.createElement('bndbox')

    createChildNode(doc, 'xmin', str(int(xmin)),
                    bndbox_node)

    createChildNode(doc, 'ymin', str(int(ymin)),
                    bndbox_node)

    createChildNode(doc, 'xmax', str(int(xmax)),
                    bndbox_node)

    createChildNode(doc, 'ymax', str(int(ymax)),
                    bndbox_node)
    object_node.appendChild(bndbox_node)


   ###------keypoints
    lm_node = doc.createElement('lm')
    num_pt=keypoints.shape[0]
    for i in range(num_pt):
        x=round(keypoints[i][0],3)
        y=round(keypoints[i][1],3)
        x_str='x'+str(i)
        y_str='y'+str(i)
        createChildNode(doc, x_str, str(x),lm_node)
        createChildNode(doc, y_str, str(y),lm_node)
    object_node.appendChild(lm_node)
    return object_node

# 将documentElement写入XML文件�?
def writeXMLFile(doc, filename):
    tmpfile = open('tmp.xml', 'w')

    doc.writexml(tmpfile, addindent='' * 4, newl='\n', encoding='utf-8')


    tmpfile.close()

    # 删除第一行默认添加的标记

    fin = open('tmp.xml')
    # print(filename)
    fout = open(filename, 'w')
    # print(os.path.dirname(fout))

    lines = fin.readlines()

    for line in lines[1:]:

        if line.split():
            fout.writelines(line)

            # new_lines = ''.join(lines[1:])

        # fout.write(new_lines)

    fin.close()

    fout.close()


# 用labelme标注： 对一个目标先标注其包围框(bbox/rectagle0, 再标注其内部的关键点（polygon)（关键点由算法工程师自己定义）
# 标注时文件结构如下
# data_dir
# data_dir/images 存放图片
# data_dir/annotations 存放json标签
# data_dir/xmlann 存放转化后的xml标签文件



def parse_args():
	parser = argparse.ArgumentParser(description="This scriptand creates database for training. ",
	                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--data_root", type=str, default='../data/original_data',
	                    help="path to database")
	parser.add_argument("--anno_save_path", type=str, default='../data/original_data/annoxml',
	                    help="path to output file")
	parser.add_argument("--is_show", type=bool, default=True,
	                    help="output image size")
	args = parser.parse_args()
	return args


if __name__ == "__main__":
    args=parse_args()
    images_path=os.path.join(args.data_root,"images")
    jsons_path=os.path.join(args.data_root,"annotations")
    fileList = os.listdir(images_path)
    _ANNOTATION_SAVE_PATH=args.anno_save_path

    if len(fileList) == 0:
        os._exit(-1)

    current_dirpath = os.path.dirname(os.path.abspath('__file__'))

    if not os.path.exists(_ANNOTATION_SAVE_PATH):
        os.mkdir(_ANNOTATION_SAVE_PATH)

    for file in fileList:
        filename = os.path.splitext(file)[0]
        filetype = os.path.splitext(file)[1]
        image_path = os.path.join(images_path, file)
        json_path = os.path.join(jsons_path, filename + ".json")

        if filetype != ".jpg":
            continue
        if not os.path.exists(json_path):
            os.remove(image_path)
            continue


        saveName = filename
        print(saveName)

        xml_file_name = os.path.join(_ANNOTATION_SAVE_PATH, (saveName + '.xml'))
        img = cv2.imread(image_path)
        height, width, channel = img.shape
        print(height, width, channel)

        my_dom = xml.dom.getDOMImplementation()

        doc = my_dom.createDocument(None, _ROOT_NODE, None)

        # 获得根节
        root_node = doc.documentElement

        # folder节点
        createChildNode(doc, 'folder', _FOLDER_NODE, root_node)

        # filename节点
        createChildNode(doc, 'filename', saveName + '.jpg', root_node)

        # source节点
        source_node = doc.createElement('source')

        # source的子节点
        createChildNode(doc, 'database', _DATABASE_NAME, source_node)
        createChildNode(doc, 'annotation', _ANNOTATION, source_node)
        createChildNode(doc, 'image', 'flickr', source_node)
        createChildNode(doc, 'flickrid', 'NULL', source_node)
        root_node.appendChild(source_node)

        # owner节点
        owner_node = doc.createElement('owner')

        # owner的子节点
        createChildNode(doc, 'flickrid', 'NULL', owner_node)
        createChildNode(doc, 'name', _AUTHOR, owner_node)
        root_node.appendChild(owner_node)

        # size节点

        size_node = doc.createElement('size')
        createChildNode(doc, 'width', str(width), size_node)
        createChildNode(doc, 'height', str(height), size_node)
        createChildNode(doc, 'depth', str(channel), size_node)
        root_node.appendChild(size_node)

        # segmented节点
        createChildNode(doc, 'segmented', _SEGMENTED, root_node)

        #读取单张图片中目标相关的信息
        labeldata = json.load(open(json_path))
        shapes = labeldata['shapes']
        num = int(len(shapes) / 2)

        for objID in range(num):
            bbox = shapes[2 * objID]
            points_b = bbox['points']
            x_min = points_b[0][0]
            y_min = points_b[0][1]
            x_max = points_b[1][0]
            y_max = points_b[1][1]
            box=(x_min,y_min,x_max,y_max)

            polygon = shapes[2 * objID + 1]
            polygon_points = polygon['points']
            pt2d_real = np.array(polygon_points)

            object_node = createObjectNode(doc, box,pt2d_real,bbox['label'])
            root_node.appendChild(object_node)

            pt1 = (int(x_min), int(y_min))
            pt2 = (int(x_max), int(y_max))
            cv2.rectangle(img, pt1, pt2, (255, 255, 0), 2)
            for id in range(pt2d_real.shape[0]):
                center = (int(pt2d_real[id][0]), int(pt2d_real[id][1]))
                cv2.circle(img, center, 2, (0, 255, 255), 2)
        if args.is_show:
            cv2.namedWindow('check', cv2.WINDOW_NORMAL)
            cv2.imwrite("bboxlm.jpg",img)
            cv2.waitKey(0)
        ### ------写入该XML文件中
        print(xml_file_name)
        writeXMLFile(doc, xml_file_name)