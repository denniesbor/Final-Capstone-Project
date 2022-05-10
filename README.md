### **Introduction**

Agriculture is an important sector in a our economy. Due to population growth, and effects of climate change there is need to adopt modern technologies to improve production, and sustain the current population. In this notebook, we apply computer vision for animal behavior monitoring and activity recoqnition. The project is divided into three stages which is object detection, classification and tracking.
![cow](https://github.com/denniesbor/Final-Capstone-Project/blob/dev/cow.png)

#### **Object detection**

Training a model from scratch is computationally expensive, time consuming, and requires lots of iteration and several architecturial tweaks to give desirable results. Transfer learning is essential for our case, and requires only finetuning to detect and classify the objects we are intrested in.

The model weights of our project are trained on COCO dataset. COCO or common objects in context is a large-scale object detection, segmentation, and captioning dataset comprosing of 90 objects.  Cows are part of this dataset, meaning the models are aware of our custom dataset. The advantage is we need little finetuning unlike building the neural network from scratch.

The models architecture comprises of convolution and dense layers optimized to give the best performance.Object detection networks are classified as multi-stage or single stage. Examples of single staged neural nets are the SSD, YOLO, etc. The multi staged approaches uses the region proposal networks in their architectures to extract feature maps from the backbone. Examples of multi stage networks are the RCNN and RFCN.
![architecure](https://github.com/denniesbor/Final-Capstone-Project/raw/dev/FRCNN.png)
[Figure 1 adapted from https://medium.com/@hirotoschwert/digging-into-detectron-2-47b2e794fabd]

We use a two stage neural network for this project, FRCNN because parameters are shared between the two stages creating an efficient framework for detection. R-CNN models is a multi layered conv neural network and consists of the feature extractor, a region proposal algorithm to generate bounding boxes, a regression and classification layer. R-CNNs tradeoff their speed for accuracy. They are generally slow on CPUs compared to single stage detectors, albeit with high accuracy.

#### **Tracking**.

The initial concept was to keep track of animal behavior by the use of memory cells such as LSTM. However this is computationally expensive and opted for the use of Simple Online Realtime Tracking algorithm(SORT). SORT allows multiple object tracking(MOT) where each object is represented by a bounding box and attempts to associate the detections across frames in a video sequence. The algorithm utilizes Kalman filter for the prediction of the next object state.
The caveat for using algorithm is occluded objects are not tracked, and will be assigned new tracking id on reentry. For our case in tracking the cattle, some of the cattle are retracked once they disappear in the field of view, or reappear.

#### **Technologies**

* [Python 3.8](https://www.python.org/downloads/release/python-380/)
* [Facebook Detectron2](https://github.com/facebookresearch/detectron2)
* [Torch](https://pytorch.org/)
* [Norfair]https://github.com/tryolabs/norfair)
* [Scikit learn](https://scikit-learn.org/)
* [Colab](https://colab.research.google.com/drive/1sqcj9646KznuejfASMoBmx1VQcMC68m_#scrollTo=Df_xYpWcY2ZY)

