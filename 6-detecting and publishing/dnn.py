from std_msgs.msg import String,Int32
import numpy as np
from numpy.random import choice,rand
import xlrd
import roslibpy
import rospy
import re
from scipy.optimize import curve_fit
import pandas as pd
from datetime import datetime
import time
from keras.models import Sequential,load_model
import os
import subprocess
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------

def mask_mean(x):
	s=x.shape[1]
	a=np.zeros(s-1)
	for i in range(0,s-1):
		a[i]=x[0,i+1]-x[0,i]
	b=np.mean(a)
	return b
def row_mean(x,step):
	s=x.shape[1]
	a=0
	b=step
	y=np.zeros(s+1-step)
	i=0
	while b<=s:
		#print(i)
		y[i]=mask_mean(x[0,a:b].reshape(1,b-a))
		a=a+1
		b=b+1
		i=i+1
	return y
def mat_mean(x,step):
	s1=x.shape[0]
	s2=x.shape[1]
	y=np.random.rand(s1,s2+1-step)
	for i in range(0,s1):
		y[i,:]=row_mean(x[i,:].reshape(1,s2),step)
	return y


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
        W[:,i]=np.round(v[24*i:24*(i+1)],1)
    file.close()
    np.round(W[0:21,:])	
    return W
def give_id(x):
  b=np.zeros(x.shape[0])
  for i in range (0,x.shape[0]):
    for j in range(0,x.shape[1]):
      if x[i,j]==1:
        b[i]=j;
        
  return b  

def my_search(a,x):
  b=np.zeros(a.shape[0])
  for i in range (0,a.shape[0]):
    for j in range(0,a.shape[1]):
      if a[i,j]==x:
        b[i]=j;
        
  return b       

def norm(mat):
	#step=5
	#mat=mat_mean(matri,step)
	
	s=mat.shape[1]
	t=np.linspace(0,s,60)
	t=t.astype(int)
	t[-1]=t[-1]-1
	xx=np.zeros([mat.shape[0],60])
	for j in range(0,mat.shape[0]):
		for i in range (0,60):
			xx[j,i]=mat[j,t[i]]	
	ss=xx.shape[0]
	y=xx.copy()
	for i in range(0,ss):
		M=np.max(xx[i,:])
		m=np.min(xx[i,:])
		y[i,:]=xx[i,:]*255/(M-m)-255*m/(M-m)
	return y
def return_list(string):
    L=[]
    i=1
    j=1
    while j<=len(string)-1:
        while not string[j] == '\n':
            j=j+1
        L.append(string[i:j])
        i=j+1
        j=i
    return L
#--------------------------------------------------------------
#--------------------------------------------------------------
#--------------------------------------------------------------
classes={0:'qalat',1:'narenji',2:'sorkh',3:'sefid',4:'siah',5:'abi',6:'qahveiy'
,7:'bale',8:'na',9:'danestan',10:'eshtebah',11:'daneshgah',12:'dars',13:'hesadat'
,14:'ghamgin'}
#danestan 458
#daneshgah 864
#dars 516
#eshtebah 476
user_dir="/home/salar-basiri/user profiles" #this is where user profiles are kept
word_dir="/home/salar-basiri/word profiles" #this is where word profiles are kept
realtime_path="/home/salar-basiri/temp" #where real-time data from the glove is stored. set temp for real-time!
model=load_model("/home/salar-basiri/last_model.HDF5",compile=False) #DNN Network
wsc_address="/home/salar-basiri/word score calculation.xlsx"
usc_address="/home/salar-basiri/user score calculation.xlsx"
RASA_IP='192.168.1.101' #IP Adress of Rasa
RASA_PORT=9090


px=norm(return_mat(realtime_path))
px=px.reshape(1,24,60,1)
pred=model.predict(px)
pred_sorted=np.sort(pred)
print(classes.get(int(give_id(np.round(pred))[0])),' : ',np.round(pred_sorted[0,-1]*100,1),'%') #this line gives the maximum predicted
if pred_sorted[0,-2] >= 0.005:
	print(classes.get(int(my_search(pred,pred_sorted[0,-2])[0])),' : ',np.round(pred_sorted[0,-2]*100,1),'%')
print('******************************************')
#a=np.array([[0,1,2,6,5,6],
#	[1,2,3,-4,5,9]])
#print(a.shape)
#print(norm(return_mat(realtime_path)))
