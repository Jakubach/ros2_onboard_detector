# Onboard Dynamic Object Detection and Tracking for Autonomous Mobile Robots for ROS2 

The repository has been forked from https://github.com/Zhefan-Xu/onboard_detector and integrated with ROS2.

## I. Introduction
This repository contains the implementation of Dynamic Obstacle Detection and Tracking (DODT) algorithm which aims at detecting and tracking dynamic obstacles for robots with extremely constraint computational resources.

If you find this work helpful, kindly show your support by giving us a free ⭐️. Your recognition is truly valued.

This repo can be used as a standalone package and also comes as a module of our [autonomy framework](https://github.com/Zhefan-Xu/CERLAB-UAV-Autonomy).

The related paper can be found on:

**Zhefan Xu\*, Xiaoyang Zhan\*, Yumeng Xiu, Christopher Suzuki, Kenji Shimada, "Onboard dynamic-object detection and tracking for autonomous robot navigation with RGB-D camera”, IEEE Robotics and Automation Letters (RA-L), 2024.** [\[paper\]](https://ieeexplore.ieee.org/document/10323166) [\[video\]](https://youtu.be/9dKX3BRnxyw).

*The authors contributed equally.



https://github.com/Zhefan-Xu/onboard_detector/assets/55560905/2f736e5d-ffbb-4e31-a440-6a5c984fa87f



## II. Installation
This package has been tested on Ubuntu 22.04 LTS with ROS Humble on [Oak-D Lite]. Make sure you have installed the compatible ROS version. 
```
# this package needs ROS2 vision_msgs package
sudo apt install ros-humble-vision-msgs

cd ~/ros2_ws/src
git clone https://github.com/Jakubach/ros2_onboard_detector.git
cd ..
colcon build
```


### b. Run on your device
Please adjust the configuration file under ```cfg/detector_param.yaml``` of your camera device. Also, change the color image topic name in ```scripts/yolo_detector/yolo_detector.py```

From the parameter file, you can find that the algorithm expects the following data from the robot:

- Depth image: ```/camera/depth/image_rect_raw```

- Robot pose (used when `localization_mode` is set to `1`): ```/mavros/local_position/pose```

- Robot odom (used when `localization_mode` is set to `0`): ```/mavros/local_position/odom```

- Color image (used when YOLO is applied): ```/camera/color/image_rect_raw```

- Aligned depth image (used when YOLO is applied): ```/camera/aligned_depth_to_color/image_raw```

```
# Launch your device first. Make sure it has the above data.
ros2 launch onboard_detector run_detector.launch
```


## V. Citation and Reference
If you find this work useful, please cite the paper:
```
@article{xu2023onboard,
  title={Onboard dynamic-object detection and tracking for autonomous robot navigation with RGB-D camera},
  author={Xu, Zhefan and Zhan, Xiaoyang and Xiu, Yumeng and Suzuki, Christopher and Shimada, Kenji},
  journal={IEEE Robotics and Automation Letters},
  volume={9},
  number={1},
  pages={651--658},
  year={2023},
  publisher={IEEE}
}
```

## VI. TODO list:
- Provide a ROS2 demo bag
- Provide a YOLO implementation from the main repository
