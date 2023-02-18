

# Iranian License Plate Detection Using Yolo-v7:



The flowchart of the model is illustrated below:

![My Image](flowchart.PNG)

Here, I applied Yolov7 in two stages

First yolo is to detect plates and crop them from the raw image.
The second yolo is to detect numbers and letters from the cropped plate.

## Stage1:

First, I clone the yolov7 from the github repository 'https://github.com/WongKinYiu/yolov7' to my directory. The pre-trained weights are also available from the same address. 

To train the yolo for plate detection, I utilized the dataset from the links below:

https://www.kaggle.com/datasets/skhalili/iraniancarnumberplate

Since the annotations are not in the right format of Yolo-v7, I used the service provided by the Roboflow website to generate the corresponding annotations for Yolo-v7.

The modified dataset is available using a couple of codes below:

## usage
```python
from roboflow import Roboflow  
rf = Roboflow(api_key="ge04UulX2BqHjBuPZwfV")   
project = rf.workspace("platedetection-jgwnf").project("plate_detection-6e2ul")   
dataset = project.version(1).download("yolov7") 
```

To download the dataset, you need to first install roboflow in your environment.

After exporting the images and annotations into the directory of the first yolov7, I finetuned the yolo-v7 using the command below:

```bash
!python train.py --batch 1 --cfg cfg/training/yolov7.yaml --epochs 30 --data you-data-path/data.yaml --weights 'yolov7.pt' --device 0 

```

## Stage2:

I created a different directory for my second Yolo.

Like the first stage, I cloned the yolov7 from the github repository 'https://github.com/WongKinYiu/yolov7' to my second  Yolo directory. The pre-trained weights are available from the same address. 

In the second Yolo, we need to detect numbers and letters on the detected license plate from the fisrt Yolo.

I applied the dataset provided in the link below for finetuning the second yolov7 to detect numbers and letters inside the plate.

https://github.com/roozbehrajabi/ALPR_Dataset/tree/main/Faster_R-CNN_dataset

Like the previous dataset, we need to change the annotation appropriate for yolov7, then regarding the previous stage, I applied the service provided by Roboflow.

After all, the modified images and annotations are available using the following codes:

```python
from roboflow import Roboflow 
rf = Roboflow(api_key="ge04UulX2BqHjBuPZwfV") 
project = rf.workspace("platedetection-jgwnf").project("numdetection")   
dataset = project.version(1).download("yolov7")  
```

After downloading the dataset, I fine-tuned the second yolov7 to detect letters and numders inside the plates.
```bash
!python train.py --batch 1 --cfg cfg/training/yolov7.yaml --epochs 30 --data ../numdetectiondata/data.yaml --weights 'yolov7.pt' --device 0 
```

Notice: For the sake of time, I only utilized 500 samples of the entire dataset for the training of the second yolo. However, to increase the performance of the model, we can use the entire dataset or add other datasets if available.

I uploaded pretrained weights of the first and second yolo in the folder '/weights'. 

## Test:
Copy the file 'yolo1/detect_paltes.py' into the directory of the first yolo and run it.  

```bash
%cd to-your-first-yolo-directory
! python detect_paltes.py
```

There are input and output directories that need to be defined by yourself. However, the default of the input and output directories are as follows:

```python
input_path = '../Input_images/'
out_path = '../outputs/detected_plates/'
```
The images of the cropped license plates will be saved into the out_path directory.

Then, in the command line, cd to the directory of the second yolo and run the command bellow:

```bash
! python detect.py --weights best.pt --conf 0.01 --source ../outputs/detected_plates
```
## Notice: 
The --source should be the directory of your generated cropped license plates. The output results will be saved into the 'runs\detect\exp5\'.


Here are some examples of the input and output of the first and second yolos:



<img src="https://github.com/MoAKgit/Iranian-License-Plate-Detection/blob/master/imges/car1.jpg" width=20% height=20%> | <img src="https://github.com/MoAKgit/Iranian-License-Plate-Detection/blob/master/imges/car1_plate.jpg"> | <img src="https://github.com/MoAKgit/Iranian-License-Plate-Detection/blob/master/imges/plate_1.jpg"> 


<img src="https://github.com/MoAKgit/Iranian-License-Plate-Detection/blob/master/imges/car2.jpg" width=20% height=20%> | <img src="https://github.com/MoAKgit/Iranian-License-Plate-Detection/blob/master/imges/cra2_plate.jpg"> | <img src="https://github.com/MoAKgit/Iranian-License-Plate-Detection/blob/master/imges/plate_4.jpg"> 


