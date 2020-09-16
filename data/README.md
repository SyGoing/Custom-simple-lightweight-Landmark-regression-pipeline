## 1、简单介绍
		本算法采用的是直接回归关键点坐标的方式。通过labelme标注生成关键点数据json.
		（1） 一般地，针对目标外轮廓关键点回归可以直接采用labeme标注目标外轮廓，然后由关键点直接生成目标的外轮廓的最小包围盒，即目标的bounding box。
		此处测试案例是车牌的外轮廓四个关键点的回归定位。外轮廓四个关键定点，标注完成则bbox也随之产生，因此没必要再重新标注bbox
        标记的结果可用于生成bbox数据用作检测任务，也可以生成样本用于直接回归四个顶点。（特殊任务）
		（2） 对于通用任务的关键点（关键点不一定是目标的外轮廓稀疏点），则最好采用专业的bbox标注工具标注目标bbox,然后由bbox裁剪出来，
		再对小图进行标注（如手势识别中的手指关键点）关键点，此时关键点的标注工具可以采用其他的工具，或者依旧是采用labelme，仅仅需要
		注意标注时候的点序。
		
 
 ## 2、项目难点（对于我来说是难点？ 数据处理这一块代码多熟悉）
    1） 代码量比较多的地方就是训练数据处理这一块，要增加对于多样化数据的支持。因为要做车牌的定位，所以选择采用labelme标注
	，标注工具上还有别的选择专门用来标注关键点（注意）。总结下来数据输入有两种形式：json(labelme)和txt
	txt数据的排列是：图像文件名称 关键点绝对坐标（x y x y x y...)

	
### 用labelme标注： 对一个目标先标注其包围框(bbox/rectagle0, 再标注其内部的关键点（polygon)（关键点由算法工程师自己定义）
* 标注时文件结构如下
 data/original_data/images 存放图片
 data/original_data/annotations 存放json标签
 data/original_data/xmlann 存放转化后的xml标签文件
 
### 按照外轮廓关键点裁剪/按照标注bbox裁剪的图像及其关键点list.txt
* 裁剪小图目录结构
 data/processed_data/croped_images
 data/processed_data/landmark_list.txt
 
 
 