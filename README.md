# TerraSense: A Terrain Classifier for Quadruped Robots
Spring 2024 Group 3 for Michigan Tech's EE/CS 5841 Machine Learning

## Overview

Successful locomotion of quadruped and bipedal robots is reliant on a good understanding of the ground surface.  While most work in the field is completed on flat indoor or outdoor surfaces, such as tiles and concrete, there are many more terrains a rugged quadruped may be expected to traverse. 

Different terrains however, have different surface conditions, and as the terrain changes, the robot must adapt it’s gait and controls to account for loose, sticky, or slippery surfaces, such as sand, dirt, snow, rocks, and grass. Neural networks can be trained on images of various terrain types to inform the robot when to adjust gait based on terrain. Furthermore, we propose that use of point cloud information in addition to standard images from a RGB depth camera can improve terrain classification accuracy.

## Implementation & Pipeline

![Alt Text](/Project_Work/artifacts/unitree_go.jpeg)

A Unitree Go1 Quadruped was used as the data collection platform, and am inexpensive, off the shelf Intel Realsense d435i depth camera was mounted looking forward.  An external camera was used to demonstrate functionality that can be applied beyond specific sensors on the Unitree Go1.  Data was collected using the Realsense ROS driver on ROS Melodic, and saved as a series of ROS bags.

Images were extracted from ROS bags using `bag2image.py`, and pointclouds were processed into .csv files of each ROS PointCloud2 message.  Each .csv row contains individual points where columns represent X, Y, Z, and Intensity of returned points.

![Alt Text](/Project_Work/artifacts/Data_Flow_Diagramme.png)

## How to get setup

The dataset collected and used in this project is available for download from our Google Drive links below.  Test/Train/Validation splits have already been created with 10% available data assigned as test data, 20% assigned to validation data, and the remaining 70% assigned to training.

### Download the dataset(s) from Google Drive
* [Image dataset](https://drive.google.com/drive/folders/1yH1uD2ji48kywCs0Q3U2Ink2GidXXb2T?usp=sharing "Image Dataset Google Drive Link")
* [Pointcloud dataset](https://drive.google.com/drive/folders/1UgTDYL5rexAGOraKXm7SAX6G8yew8W7J?usp=sharing "Pointcloud Dataset Google Drive Link")

### Install the Repo
`git clone https://github.com/iqmattso/5841_TerraSense.git` 

Dependacies:
* CUDA
* PyTorch
* Numpy
* MatplotLib
* Seaborn

## References
[1] W. Wang et al., “A visual terrain classification method for mobile robots’ navigation based on convolutional neural network and support vector machine,” Transactions of the Institute of Measurement and Control, vol. 44, no. 4, pp. 744–753, Feb. 2021. doi:10.1177/0142331220987917 

[2] A. Chilian and H. Hirschmuller, “Stereo camera based navigation of mobile robots on Rough Terrain,” 2009 IEEE/RSJ International Conference on Intelligent Robots and Systems, Oct. 2009. doi:10.1109/iros.2009.5354535

[3] A. Kurup, S. Kysar, J. Bos, P. Jayakumar, and W. Smith, “Supervised terrain classification with adaptive unsupervised terrain assessment,” SAE International Journal of Advances and Current Practices in Mobility, vol. 3, no. 5, pp. 2337–2344, Apr. 2021. doi:10.4271/2021-01-0250 

[4] R. Q. Charles, H. Su, M. Kaichun, and L. J. Guibas, “PointNet: Deep Learning on point sets for 3D classification and segmentation,” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Jul. 2017. doi:10.1109/cvpr.2017.16 

## Contacts
* Ian Q. Mattson, iqmattso@mtu.edu
* Anders Smitterberg, jasmitte@mtu.edu
* Satyanarayana Velamala, svelamal@mtu.edu