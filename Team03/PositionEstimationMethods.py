# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 18:32:28 2022

@author: deshe
"""
import numpy as np
import random 

image_width = 48000 #meters
pixel_Width = 248 #pixels
Focal_Length = 543.45 #mm

def Position_Estimation(Focal_Length, real_image_width, image_width_in_pixels):
    distance = (real_image_width * Focal_Length)/image_width_in_pixels
    return distance

def xy_distance(theta,Image_distance):
    phi = (90-theta)*(np.pi/180)
    x_distance = abs(np.cos(phi)*Image_distance)
    y_distance = abs(np.tan(phi)*Image_distance)
    return x_distance,y_distance


pixel_width = np.linspace(50,500, num = 100)
Image_dist_list = []
Image_distance = 100000
for pixels in pixel_width:
    if Image_distance > 45000:
        percent_change = int((1 * pixels)/100)
        num = np.random.normal(-percent_change, percent_change)
        Image_distance = Position_Estimation(Focal_Length, image_width, pixels)
        Image_dist_list.append(Image_distance+num)
    else:
        print("WARNING! Distance is less than 45 km from target HLS")
       
theta = np.linspace(0,45, num = 46)
XY_distance = [[],[]]
print (Image_distance)
for i in theta:
    distance = xy_distance(i, Image_distance)
    XY_distance[0].append(distance[0])
    XY_distance[1].append(distance[1])
    
print(XY_distance[1])
