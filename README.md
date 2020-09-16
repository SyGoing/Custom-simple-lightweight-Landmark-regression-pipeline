# Custom simple lightweight landmark regression task pipeline
	In this repo, I have provided the pipeline of landmark regression task,which consists of the dataset making,training,
	test and export to onnx for applications not only for face landmark regression.
## datasets
	
### mark keypoints 
 * For some outline keypoints regression tasks, you could just mark it py the [labelme](https://github.com/wkentaro/labelme). Actually, the object's bbox could be create by the outline points.
 * For other keypoints, you'd best mark the bbox of the object firstly,and mark the keypoints on the bbox croped image. Anyway,you could also mark bbox and 
 keypoints of the object simultaneously by the labelme( first bbox,then keypoints(polygon)) or other tools. In this repo, I have used the labelme for marking
 the bbox and the four vertexes by the labelme.
 
### convert outline style landmarks to the list.txt [path/image x1 y1 x2 y2 ....] 
	cd ./datasets
	1) only marked the outline points
	   python create_plate_data.py
    2) bbox and keypoints have been marked
       python create_box_points_data.py
	   
### convert labelme json format to voc style for multitask training if you need
	cd ./datasets 
	python labelmeJson2VOC.py
	
## training
   For landmark regression, the wingloss has been used. The model's design refered to pfld and Onet. I have used the Onet for car plate vertexes regression.
   train.py
## Inference
   demo.py
## export to onnx
   pytorch2onnx.py

## Reference
	* [pfld-pytorch](https://github.com/polarisZhao/PFLD-pytorch)
	* [wingloss paper](https://arxiv.org/abs/1711.06753)
	
	
	
   
	
	
  
