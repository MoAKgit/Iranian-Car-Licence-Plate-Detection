

# Iranian Licence Detection Using Yolo-v7:


The flowchart of the model is illustrated as bellow:




Here, I applied Yolov7 in two stages 
1) For detecting plates and cropping them from the vehicle
2) Detecting numbers and letters from the croped plate.



stage1:


In order to train the yolo for plate detection, I used two datasets from the links bellow:


After combinging these two dataset, Since the annotations are not in the right format of Yolo-v7,
 I used the service provided by Roboflow website to generate corresponding annotations for Yolo-v7.


The dataset is available using theses lines of code:

from roboflow import Roboflow
rf = Roboflow(api_key="ge04UulX2BqHjBuPZwfV")
project = rf.workspace("platedetection-jgwnf").project("plate_detection-6e2ul")
dataset = project.version(1).download("yolov7")

To download the dataset, you need to first install roboflow in your envirenment.




After exporting the images and annotations, I finetuned the yolo-v7 using the command bellow:

Notice that yolo-v7 should be clone from the Github:

and also the pretrained weights are available from the link bellow:



#######################################################################
stage2:

In the second stage, we need to detect numbers and letters in the plate, so then inspired py the work:

I applied the dataset provided in the link bellow for funetuning the second yolov7 in order to detect numbers and letters inside the plate.



like the previous dataset, we need to change the annotation approperiate for yolov7, then regarding the previous stage, I applied the service provided by roboflow.


After all, the modified images and annotations are available using the following codes:

from roboflow import Roboflow
rf = Roboflow(api_key="ge04UulX2BqHjBuPZwfV")
project = rf.workspace("platedetection-jgwnf").project("numdetection")
dataset = project.version(1).download("yolov7")


After downloading the dataset, I funetined the second yolov7 to detect letters inside the plates.


Notice: For the sake of time, I only utilized 500 samples of the entire dataset for the training the second yolo. 
However, to increase the performence of the model, we can use the entire dataset or add other datasets if available.















