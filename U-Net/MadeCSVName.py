#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 22:11:28 2018

@author: zgfly
"""

#made GlandsImage.csv
import csv
import os


#made GlandsImage.csv
writer=csv.writer(open('GlandsImage.csv','w',encoding='utf8',newline=''))
path="Segment/Image"
writer.writerow(["filename"])
    
for img in os.listdir(path):
    if img!='DS_Store':
        name=os.path.splitext(img)
        img_segment=name[0]
        
    root_path="Segment/Image/"
    end_path=".jpg"
    data=root_path+img_segment+end_path
    writer.writerow([data])
    
    

"""
#made GlandsMask.csv
writer=csv.writer(open('GlandsMask.csv','w',encoding='utf8',newline=''))
path="Segment/Mask"
writer.writerow(["filename"])
    
for img in os.listdir(path):
    if img!='DS_Store':
        name=os.path.splitext(img)
        img_segment=name[0]
        
    root_path="Segment/Mask/"
    end_path=".jpg"
    data=root_path+img_segment+end_path
    writer.writerow([data])
"""