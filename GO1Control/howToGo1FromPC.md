# Connecting to and controlling GO1 from a PC 

This document describes the method used to connect to a unitree go1 from a PC and run the example high level control document. This was tested working on a Ubuntu 18.04 laptop running ROS1 Melodic. These steps should all be followed in order of appearance.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
---

## Installing and configuring ROS1 Melodic

ROS installation was performed per the [official ROS wiki](https://wiki.ros.org/melodic/Installation/Ubuntu) however the specific instructions followed will be detailed here.

You may need to [configure the ubuntu repositores](https://help.ubuntu.com/community/Repositories/Ubuntu) to allow restricted, universe, and multiverse. This was unneccesary on the PC I used but it was required on another one so just FYI, check if unsure. 

### Setup Sources
Run this to set up the appropriate repositories
`sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'`

### Set Up Keys
Run this to set up the appropriate Keys
`sudo apt install curl # if you haven't already installed curl`
`curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -`

### Installation
Run all the generic apt-get commands
`sudo apt-get update`
`sudo apt-get upgrade`

Now do a full installation of ROS, this should take a few minutes
`sudo apt install ros-melodic-desktop-full`

### Environment Setup

you can run `source /opt/ros/melodic/setup.bash` every time you open the terminal, or you can add it to the bashrc file, which is what I did. I think this can cause conflicts with other versions of ROS

If you want it to source ROS environment variables every time a new terminal starts, which I did, run these commands
`echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc`
`source ~/.bashrc`

### Install dependancies for package building
To install ROS package building dependanceis run this
`sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential`

Then run these
`sudo apt install python-rosdep`
`sudo rosdep init`
`rosdep update`

Done. (hopefully)

---

## Sourcing and building all Unitree ROS packages

Most of these instructions based off of go1 documentation from [Trossen Robotics](https://docs.trossenrobotics.com/unitree_go1_docs/), however there are several typos in the documentation and it will not work if followed directly. 

The instructions followed, in their corrected form and order are listed below, this again, only works on 18.04 and ROS1 Melodic

### Setting up the workspace and cloning the unitree repos
First, a catkin workspace is created, and the unitree repos are sourced.
The unitree repos used are:

- [unitree legged sdk](https://github.com/unitreerobotics/unitree_legged_sdk)
- [unitree ros to real](https://github.com/unitreerobotics/unitree_ros_to_real)

The workspace is setup in default directory:

```
mkdir -p catkin_ws/src
cd ~/catkin_ws/src
git clone -b v3.8.0 https://github.com/unitreerobotics/unitree_legged_sdk
git clone -b v3.8.0 https://github.com/unitreerobotics/unitree_ros_to_real
```

Then ROS dependencies are gotten:

```
cd ~/catkin_ws
rosdep update --include-eol-distros
rosdep install --from-paths src --ignore-src -r -y --rosdistro=$ROS_DISTRO
catkin_make
```

Then the workspace is built:

```
cd ~/catkin_ws/
catkin_make
```

Then the examples are sourced:

`source ~/catkin_ws/devel/setup.bash`

This can be added to bashrc like the previous source file, or ran again every time the terminal is open. I added it to the bashrc file like so:

`echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc`
`source ~/.bashrc`

Done. (hopefully)

## Connecting PC to GO1 Network
This is the 

