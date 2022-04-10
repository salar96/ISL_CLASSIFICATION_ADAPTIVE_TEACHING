# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:14:47 2019

@author: Salar Basiri
"""
import numpy as np

def return_mat(address):

    file=open(address,'r+')
    s=file.read()
    #%%
    SS=s.replace('"','')
    SSS=SS.replace('data:','')
    SSS=SSS.replace('\\','')
    SSS=SSS.replace('\n','')
    SSS=SSS.replace('---','')
    S=SSS.split(" ")
    q=[x for x in S if x != '']
    v=[float(x) for x in q]
    #%%
    frame_num=int(len(v)/24)
    W=np.random.rand(24,frame_num)
    for i in range(0,frame_num):
        W[:,i]=v[24*i:24*(i+1)]
    file.close()
    return W

#%%
import os
path=r"C:\Users\Salar Basiri\Desktop\data new 2 - Copy"

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        files.append(os.path.join(r, file))

mats=[]
for i in files:
    mat=return_mat(i)
    add1=r"C:\Users\Salar Basiri\Desktop\matsddd"
    add2=""    
    add3=i[i.find('data'):]
    add4=".npy"
    add=add1+add2+add3+add4
    add=add.replace('.txt.','.')
    np.save(add,mat)

