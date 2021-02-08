# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 22:53:34 2021

@author: lsong4
"""

import os
import csv 

# path = 'dataset/1_enhanced/'
# fileList = os.listdir(path)

# n=0
# for i in fileList:
    
#     oldname=path+ os.sep + fileList[n]
    
#     newname=path + os.sep + 'real' + str(n+1)+'.jpg'
    
#     os.rename(oldname,newname)
#     print(oldname,'======>',newname)
    
#     n+=1
    
path = 'dataset/all/'
fileList = os.listdir(path)

  
# opening the csv file in 'w+' mode 
file = open('./dataset/hornet_label.csv', 'w+', newline ='') 
  
# writing the data into the file 
with file:     
    write = csv.writer(file) 
    write.writerows(fileList) 