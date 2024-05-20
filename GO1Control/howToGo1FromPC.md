# Connecting to and controlling GO1 from a PC 

This document describes the method used to connect to a unitree go1 from a PC and run the example high level control document. This was tested working on a Ubuntu 18.04 laptop running ROS1 Melodic. These steps should all be followed in order of appearance.

---

## Installing and configuring ROS1 Melodic

ROS installation was performed per the [official ROS wiki](https://wiki.ros.org/melodic/Installation/Ubuntu) however the specific instructions followed will be detailed here.

You may need to [configure the ubuntu repositores](https://help.ubuntu.com/community/Repositories/Ubuntu) to allow restricted, universe, and multiverse. This was unneccesary on the PC I used but it was required on another one so just FYI, check if unsure. 

### Setup Sources
Run this to set up the appropriate repositories
`sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'`

### Set Up Keys
Run this to set up the appropriate Keys
`sudo apt install curl`
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

## Connecting PC to GO1  Internal Network
This is the portion where the PC needs to connect to the network switch of the GO1. You can reference [this document](https://docs.trossenrobotics.com/unitree_go1_docs/getting_started/network.html), however there are a few typos, and the specific instructinos followed have been detailed here:

This is useful for this task as well as "SSH-ing" and other utilities

Start by connecting a ubuntu 18.04 laptop and the GO1 together with a ethernet cable.

Ensure the GO1 is fully powered on.

Now it will almost certainly refuse the connection.

Open settings and see if it is possible to determine what ethernet device id the laptop's ethernet port is.

It may be something like "enp5s0" or something along those lines; it should at least start with "en". If there are multiple, some trial and error may have to be employed to determine the correct one.

Now open a terminal.

Type `ifconfig`
You may need to install ifconfig if it is not already installed

it will spit out a list of all the interfaces on the computer and their status

Something like this:
```
enp5s0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 192.168.0.193  netmask 255.255.255.0  broadcast 192.168.0.255

wlp0s20f3: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 192.168.0.183  netmask 255.255.255.0  broadcast 192.168.0.255
```

In the case of this example `enp5s0` is the port the go1 is connected to

Run the following code to configure the port. Replace `enp5s0` with whatever the port is on the specific machine you are running

`sudo ifconfig enp5s0 down`
`sudo ifconfig enp5s0 192.168.123.162/24`
`sudo ifconfig enp5s0 up`

This turns off the port, configures a static IP, then turns the port back on

Now test the configuration by pinging the onboard rPi.

`ping -c 3 192.168.123.161`

Done. (hopefully)

### SSH into specific computer
This is not required for running the examples, but it may be neccesary to SSH into the various onboard computers on the GO1.

a [Network layout diagram](https://unitree.droneblocks.io/learning/go1-system-architecture) can be viewed here for reference, detailing the IPs of the various onboard computers.

To ssh into a computer on the go1 determine the IP from the network layout diagram and ensure the above steps regarding connecting to the internal network are followed. 

For logging into the onboard rPi
```
ssh pi@192.168.123.161 # PW is 123
```

For logging into the onboard Nanos
```
ssh unitree@192.168.123.xx # PW is 123, xx can be 13, 14, or 15 depending on which nano it is desired to connect to
```

## Running an Example
Now that the robot is connected to the PC and powered on we can run the example code (hopefully).

Open a new terminal and run this line, which launches the high level UDP connection node for GO1

`roslaunch unitree_legged_real real.launch ctrl_level:=highlevel`

Now open a new terminal window and run the state_sub demo

`rosrun unitree_legged_real state_sub`

Now run the example_walk demo, the robot will move during this demo, ensure it is in area with plenty or space and standing up

`rosrun unitree_legged_real example_walk`

The robot should walk around for a little while and demonstrate some capabilities

The example runs for about a minute or less.

Success (hopefully)