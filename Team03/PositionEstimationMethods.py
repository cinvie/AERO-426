# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 18:32:28 2022

@author: deshe
"""
import numpy as np
import random 


# Image_width, Focal_Length and theta are all user inputs from sim
# Pixel Width is based on the 

image_width = 48000 #meters 
#image_width = 40 #meters
pixel_Width = 248 #pixels
Focal_Length = 543.45 #mm
#Focal_Length = 1000 #mm
theta = 25 #degrees

def Position_Estimation(Focal_Length, real_image_width, image_width_in_pixels):
    distance = (real_image_width * Focal_Length)/image_width_in_pixels
    return distance

def xy_distance(theta,Image_distance):
    phi = (90-theta)*(np.pi/180)
    x_distance = abs(np.cos(phi)*Image_distance)
    y_distance = abs(np.sin(phi)*Image_distance)
    return x_distance,y_distance


pixel_width = np.linspace(50,500, num = 50)
Image_dist_list = []
XY_distance = [[],[]]
Image_distance = 100000
for pixels in pixel_width:
    if Image_distance > 15000:
        percent_change = float((0.1 * pixels)/100)
        num = round(random.uniform(-percent_change, percent_change),2)
        pixel_num = pixels + num
        Image_distance = Position_Estimation(Focal_Length, image_width, pixel_num)
        x_y_distance = xy_distance(theta, Image_distance)
        Image_dist_list.append(Image_distance)
        print("num:",num)
        print("Distance to HLS from target Image:",Image_distance,"km")
        XY_distance[0].append(x_y_distance[0])
        XY_distance[1].append(x_y_distance[1])
        print("Distance to HLS from target Image (X Coordinate):",x_y_distance[0],"km")
        print("Distance to HLS from target Image (Y Coordinate):",x_y_distance[1],"km")
    else:
        print("WARNING! Distance is less than 15 km from target HLS")



