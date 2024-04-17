#!/usr/bin/env python3
import rospy
import csv
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

msg_counter = 0

def callback(data):

    #print("callback entered")
    global msg_counter

    msg_counter = msg_counter + 1
    filename = "{}.csv".format(msg_counter)
    print(filename)
    gen = point_cloud2.read_points(data)
    assert isinstance(data, PointCloud2)
    #print(type(gen))
    with open (filename, 'wt') as out:
        csv_out=csv.writer(out)
        for p in gen:
            csv_out.writerow(p)

    


def listener():

    
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/camera/depth/color/points", PointCloud2, callback)
    rospy.spin()



if __name__ == '__main__':
    listener()