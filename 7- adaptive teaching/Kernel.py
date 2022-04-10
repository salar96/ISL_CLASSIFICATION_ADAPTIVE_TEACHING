#!/usr/bin/env/python
#"In Coding We Believe..."
# Salar Basiri
# M.Sc Mechatronics
# Sharif University of Technology
# Tehran, Iran
# 2019-2020
#----------------------------------
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
def fitter(X, a, b, c,ap,bp,cp,d,e,f,g):
    x,y,z = X
    return a*x**2+b*y**2+c*z**2+ap*x+bp*y+cp*z+d+e*x*y+f*x*z+g*y*z
def give_R_word(rep,acu,spd):
    R=c[0]*rep**2+c[1]*acu**2+c[2]*spd**2+c[3]*rep+c[4]*acu+c[5]*spd+c[6]+c[7]*rep*acu+c[8]*rep*spd+c[9]*acu*spd
    if R>100:
        R=100
    if R<0:
        R=0
    return R
def word_fuzzify(R):
    if R<=25:
        alpha_w=1
        beta_w=0
        gamma_w=0
    if R>=75:
        alpha_w=0
        beta_w=0
        gamma_w=1
    if 25<=R<=50:
        gamma_w=0
        alpha_w=1-(R-25)/25
        beta_w=(R-25)/25
    if 50<=R<=75:
        alpha_w=0
        beta_w=1-(R-50)/25
        gamma_w=(R-50)/25   
    print("This word is {}% easy, {}% normal and {}% hard".format(np.round(alpha_w*100,2),np.round(beta_w*100,2),np.round(gamma_w*100,2)))
    return alpha_w,beta_w,gamma_w

def give_R_user(rep,acu,spd):
    R=c_u[0]*rep**2+c_u[1]*acu**2+c_u[2]*spd**2+c_u[3]*rep+c_u[4]*acu+c_u[5]*spd+c_u[6]+c_u[7]*rep*acu+c_u[8]*rep*spd+c_u[9]*acu*spd
    if R>100:
        R=100
    if R<0:
        R=0
    return R
def user_fuzzify(R):
    if R<=25:
        alpha_u=1
        beta_u=0
        gamma_u=0
    if R>=75:
        alpha_u=0
        beta_u=0
        gamma_u=1
    if 25<=R<=50:
        gamma_u=0
        alpha_u=1-(R-25)/25
        beta_u=(R-25)/25
    if 50<=R<=75:
        alpha_u=0
        beta_u=1-(R-50)/25
        gamma_u=(R-50)/25   
    print("This user is {}% weak, {}% normal and {}% strong".format(np.round(alpha_u*100,2),np.round(beta_u*100,2),np.round(gamma_u*100,2)))
    return alpha_u,beta_u,gamma_u

def Gen_out_FRB(id,alpha_w,beta_w,gamma_w,alpha_u,beta_u,gamma_u): #generate output based on fuzzy rule base
    # if id is '0' means we want rep and spd, if its 'p' we want
    # aff and the word is passed, if its 'f' we want aff and 
    # the word is failed.
    p1=alpha_w*alpha_u # word easy user weak
    p2=alpha_w*beta_u  # word easy user medium
    p3=alpha_w*gamma_u # word easy user strong
    p4=beta_w*alpha_u  # word medium user weak
    p5=beta_w*beta_u   # word medium user medium
    p6=beta_w*gamma_u  # word medium user strong
    p7=gamma_w*alpha_u # word hard user easy
    p8=gamma_w*beta_u  # word hard user medium
    p9=gamma_w*gamma_u # word hard user strong   
    P=[p1,p2,p3,p4,p5,p6,p7,p8,p9]
    #--------------------------------------------
    if id=='0':
	    S=[SPEED[1],SPEED[2],SPEED[2],SPEED[0],SPEED[1],SPEED[2],SPEED[0],SPEED[0],SPEED[1]]
	    R=[REPEAT[1],REPEAT[0],REPEAT[0],REPEAT[2],REPEAT[1],REPEAT[0],REPEAT[2],REPEAT[2],REPEAT[1]]
	    out_spd=np.round(np.average(S,weights=P),2)
	    out_repeat=np.average(R,weights=P)
	    out_repeat=np.round(4/9*out_repeat+5/9) # to make it from 1 to 5
	    print("For the given user and word, speed should be {} (out of 10) and repeat# should be {}.".format(out_spd,out_repeat))
	    return out_spd,out_repeat
    elif id=='p':
	    A=[AFF[2],AFF[1],AFF[1],AFF[2],AFF[2],AFF[1],AFF[2],AFF[2],AFF[2]]
	    out_emo=np.round(np.average(A,weights=P))
	    print("word {} Emotion should be {}".format(id,out_emo))
	    return out_emo 
    elif id=='f':
	    A=[AFF[1],AFF[0],AFF[0],AFF[1],AFF[1],AFF[0],AFF[2],AFF[1],AFF[1]]
	    out_emo=np.round(np.average(A,weights=P))
	    print("word {} Emotion should be {}".format(id,out_emo))
	    return out_emo 
    else:
    	print("FATAL ERROR!")

def return_stat(id,alpha,beta,gamma):
    i=np.max([alpha,beta,gamma])
    if alpha==i:
        if id=='w':
            stat='EASY'
        if id=='u':
            stat='WEAK'
    if beta==i:
            stat='MEDIUM'
    if gamma==i:
        if id=='w':
            stat='HARD'
        if id=='u':
            stat='STRONG'
    return stat

def write_session(id,file,n,name,date,r,stat):
    file.write("--------------------\n")
    file.write("SESSION {}:\n".format(n))
    if id=='w':
        file.write("USER = {}\n".format(name))
    else:
      file.write("WORD = {}\n".format(name))  
    file.write("DATE = {}\n".format(date))
    file.write("\nThe teaching will be based on the WSCM Module:\n")
    file.write("\nRATING = {}\n".format(r))
    file.write("STAT = {}\n".format(stat))
def write_performance(file,rep,acu,spd):
    file.write("\nThe user performance at the end of the section:\n")
    file.write("\nrep = {}\n".format(rep))
    file.write("acu = {}\n".format(acu))
    file.write("spd = {}\n".format(spd))
    file.close()
def write_pass_fail(path,id,word_name):
    file=open(path,'r+')
    P_F_I=[]
    contents = file.readlines()
    for i in range(0,len(contents)):
        if contents[i]=='_ _ _\n':
            P_F_I.append(i)
    if id=='pass':
        contents.insert(P_F_I[0],word_name+'\n')
    elif id=='fail':
        contents.insert(P_F_I[1],word_name+'\n')
    else:
        print('Wrong id!couldnt write')
    file.close()
    file =open(path, "r+")
    #print(contents)
    file.writelines(contents)
    file.close()
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
user_dir="/home/salar-basiri/user profiles" #this is where user profiles are kept
word_dir="/home/salar-basiri/word profiles" #this is where word profiles are kept
realtime_path="/home/salar-basiri/temp" #where real-time data from the glove is stored. set temp for real-time!
model=load_model("/home/salar-basiri/last_model.HDF5",compile=False) #DNN Network
wsc_address="/home/salar-basiri/word score calculation.xlsx"
usc_address="/home/salar-basiri/user score calculation.xlsx"
RASA_IP='192.168.1.101' #IP Adress of Rasa
RASA_PORT=9090
# first we have to check if the hand imit topic is published correctly!
z=0
while z==0:
	print("\n\nThe Rightimit topic does not seem to be published yet! Make sure the glove is sending the data.")
	cmd = [ 'rostopic', 'list']
	output = str(subprocess.Popen( cmd, stdout=subprocess.PIPE ).communicate()[0])
	z=output.count("RightImit")
	input("Is it published now?")
print("\nYes!We have the topic now!\n")
## ok we make sure it's published
#
#

# some code here to make sure we are connected to rasa

RASA = roslibpy.Ros(host=RASA_IP, port=RASA_PORT)
print('\nNow connecting to RASA...\n')
while True:
	try:
		RASA.run()
		break
	except:
		print('\nCant connect to RASA! Please make sure that RASA is running websocket, or check the IP again!\n')
print('\nConnected to RASA Successfully!\n')
while True:
    try:
        username=input("Please enter the user name:")
        user_path=user_dir+"/"+username+".txt"
        user_file=open(user_path,'r+')
        user_file_a=open(user_path,'a')
        break
    except:
        print("Error: This user does not exist!\n Try creating this user or type the name again.\n")
print('**************************************\n')
print('The teaching will begin for {}\n.'.format(username))

while True: # this is the main teaching loop. this loop is run for every word.
	user_file=open(user_path,'r+')
	user_file_a=open(user_path,'a')

	f=user_file.read()
	########## in this section we'll see which words have been failed and which ones have been passed.
	iter_start_list=[m.start() for m in re.finditer('___', f)]
	iter_finish_list=[m.start() for m in re.finditer('_ _ _', f)]
	Pass=f[iter_start_list[0]+3:iter_finish_list[0]]
	Fail=f[iter_start_list[1]+3:iter_finish_list[1]]
	Pass_list=[int(m) for m in return_list(Pass)]
	Fail_list=[int(m) for m in return_list(Fail)]
	print('for {} these words have been passed:\n'.format(username))
	Pass_list_word=[classes.get(i) for i in Pass_list]
	print(Pass_list_word)
	print('And these words have been failed:\n')
	Fail_list_word=[classes.get(i) for i in Fail_list]
	print(Fail_list_word)
	LoW=[i for i in range(0,15)]
	weights=[1 for i in range(0,15)]
	for n in Pass_list:
	    weights[n]=weights[n]-0.5
	for n in Fail_list:
	    weights[n]=weights[n]+0.5
	for i in weights:
		if i>3:
			i=3
		if i<0.5:
			i=0.5
	r=choice(LoW,1,weights)
	#r=8
	#-----------------------------------------------------------------
	
	word=classes.get(int(r))
	print('Based on the user and word profiles:\n')
	print('The chosen word to teach is {}.\n'.format(word))
	print("So the user is: {} and the word is: {}\n".format(username,word))

	###########in this section we'll check the chosen word
	word_path=word_dir+"/"+word+".txt"

	while True:
		try:
			word_file=open(word_path,'r+')  #this file is used for reading purposes
			word_file_a=open(word_path,'a') #this file is used to append texts
			break
		except:
			input("\nCant find the word: {}! try to create it first!\n".format(word))

	g=word_file.read()
	word_session_number=g.count("SESSION")
	word_is_first = 0
	if word_session_number == 0:
	    word_is_first = 1

	rep_list_w=[m.start() for m in re.finditer('rep', g)]
	acu_list_w=[m.start() for m in re.finditer('acu', g)]
	spd_list_w=[m.start() for m in re.finditer('spd', g)]
	enter_list_w=[m.start() for m in re.finditer('\n', g)]

	if not word_session_number==0:
	    rep_w=[]
	    acu_w=[]
	    spd_w=[]
	    for i in range(0 , word_session_number):
	        j=rep_list_w[i]+6
	        while not g[j] == '\n':
	            j = j+1
	        rep_w.append(g[rep_list_w[i]+6:j])

	    rep_w=[float(i) for i in rep_w]
	    

	    for i in range(0 , word_session_number):
	        j=acu_list_w[i]+6
	        while not g[j] == '\n':
	            j = j+1
	        acu_w.append(g[acu_list_w[i]+6:j])

	    acu_w=[float(i) for i in acu_w]
	    

	    for i in range(0 , word_session_number):
	        j=spd_list_w[i]+6
	        while not g[j] == '\n':
	            j = j+1
	        spd_w.append(g[spd_list_w[i]+6:j])

	    spd_w=[float(i) for i in spd_w]
	    
	    ave_rep_w=np.average(rep_w)
	    ave_acu_w=np.average(acu_w)
	    ave_spd_w=np.average(spd_w)
	    print("____________________________")
 
	    print("The word: {} has been trained in {} sessions.\n".format(word,word_session_number))
	    print("average repeat number for this word so far: {}\n".format(ave_rep_w))
	    print("average accuracy for this word so far: {}\n".format(ave_acu_w))
	    print("average speed for this word so far: {}\n".format(ave_spd_w))
	    print("____________________________")

	else:
	    print("This word has not been trained by any user yet.\n")
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# now that we know the word, we have to calculate it's rank
	df = pd.read_excel(wsc_address, sheet_name=0)
	word_cri = df.as_matrix() # this is the main kernel all the decisions are made.
	x=word_cri[:,0]
	y=word_cri[:,1]
	z=word_cri[:,2]
	X=word_cri[:,:3]
	rix=word_cri[:,3]

	c,cov=curve_fit(fitter, (x,y,z), rix, [5,1,1,0,1,2,8,1,1,1])
	if not word_is_first:
	    word_rate=give_R_word(ave_rep_w,ave_acu_w,ave_spd_w)
	else:
	    word_rate=50
	print("word rating is: {}\n".format(word_rate))
	alpha_w,beta_w,gamma_w=word_fuzzify(word_rate)
	print("writing new information on the word profile...\n")
	write_session('w',word_file_a,word_session_number+1,username,datetime.now(),word_rate,return_stat('w',alpha_w,beta_w,gamma_w))        
	#-----------------------------------------------------------------
	#-----------------------------------------------------------------
	#-----------------------------------------------------------------

	user_session_number=f.count("SESSION")
	user_is_first = 0
	if user_session_number == 0:
	    user_is_first = 1
	rep_list_u=[m.start() for m in re.finditer('rep', f)]
	acu_list_u=[m.start() for m in re.finditer('acu', f)]
	spd_list_u=[m.start() for m in re.finditer('spd', f)]
	enter_list_u=[m.start() for m in re.finditer('\n', f)]
	if not user_session_number==0:
	    rep_u=[]
	    acu_u=[]
	    spd_u=[]
	    for i in range(0 , user_session_number):
	        j=rep_list_u[i]+6
	        while not f[j] == '\n':
	            j = j+1
	        rep_u.append(f[rep_list_u[i]+6:j])

	    rep_u=[float(i) for i in rep_u]
	    

	    for i in range(0 , user_session_number):
	        j=acu_list_u[i]+6
	        while not f[j] == '\n':
	            j = j+1
	        acu_u.append(f[acu_list_u[i]+6:j])

	    acu_u=[float(i) for i in acu_u]
	    

	    for i in range(0 , user_session_number):
	        j=spd_list_u[i]+6
	        while not f[j] == '\n':
	            j = j+1
	        spd_u.append(f[spd_list_u[i]+6:j])

	    spd_u=[float(i) for i in spd_u]
	    
	    ave_rep_u=np.average(rep_u)
	    ave_acu_u=np.average(acu_u)
	    ave_spd_u=np.average(spd_u)
	    print("____________________________\n")
 
	    print("The user: {} has been trained in {} sessions.\n".format(username,user_session_number))
	    print("average repeat number for this user so far: {}\n".format(ave_rep_u))
	    print("average accuracy for this user so far: {}\n".format(ave_acu_u))
	    print("average speed for this user so far: {}\n".format(ave_spd_u))
	    print("____________________________\n")

	else:
	    print("This user has not been trained yet!\n")
	df2 = pd.read_excel(usc_address, sheet_name=0)
	user_cri = df2.as_matrix() # this is the main kernel all the decisions are made.
	xu=user_cri[:,0]
	yu=user_cri[:,1]
	zu=user_cri[:,2]
	Xu=user_cri[:,:3]
	ru=user_cri[:,3]

	c_u,cov_u=curve_fit(fitter, (xu,yu,zu), ru, [5,1,1,0,1,2,8,1,1,1])
	if not user_is_first:
	    user_rate=give_R_user(ave_rep_u,ave_acu_u,ave_spd_u)
	else:
	    user_rate=50
	print("user rating is: {}".format(user_rate))
	alpha_u,beta_u,gamma_u=user_fuzzify(user_rate)
	print("writing new information on the user profile...\n")
	write_session('u',user_file_a,user_session_number+1,word,datetime.now(),user_rate,return_stat('u',alpha_u,beta_u,gamma_u))        

	#-----------------------------------------------------------------
	# now all these 6 fuzzy factors will go to the regulator module to decide for the robot outputs
	#Here we define 3 values for speed,repeat and affection. lowest, middle and highest.
	SPEED=[1,5,10]
	REPEAT=[1,5,10]
	AFF=[1,2,3]
	print("____________________________")
	out_spd,out_repeat=Gen_out_FRB('0',alpha_w,beta_w,gamma_w,alpha_u,beta_u,gamma_u)
	user_file_a.write("****\n")
	user_file_a.write("{} here is {} ({})\n".format(word,word_rate,return_stat('w',alpha_w,beta_w,gamma_w)))
	user_file_a.write("\nRASA WILL DO IT {} TIMES WITH THE TIMING = {}.\n".format(out_repeat,out_spd))
	word_file_a.write("****\n")
	word_file_a.write("{} here is {} ({})\n".format(username,user_rate,return_stat('u',alpha_u,beta_u,gamma_u)))
	word_file_a.write("\nRASA WILL DO IT {} TIMES WITH THE TIMING = {}.\n".format(out_repeat,out_spd))
	print("\n\n")
	#-----------------------------------------------------------------
	#-----------------------------------------------------------------
	#
	# ok in this session, we have to write a code that sends parameters to rasa



	talker = roslibpy.Topic(RASA, '/rasa_output', 'std_msgs/String')
	emo_talker=roslibpy.Topic(RASA, '/RasaFaceII', 'std_msgs/String')
	finish_talker=roslibpy.Topic(RASA, '/command_torque', 'std_msgs/Int32')
	SSS=[]
	ACC=[]
	per_rep=[]
	step=0
	word_stat='fail'


	while True:
		if step>=out_repeat:
			break
		print('ok! Lets go for the try # {}!\n'.format(step+1))
		opl=0
		while True:
			if opl==0:
				temptime=time.time()
				opl=1
				while time.time()-temptime < 1:
					talker.publish(roslibpy.Message({'data': 'User={} Word=#{}# rep={} spd={}'.format(username,word,out_repeat,out_spd)}))
					print('Sending message...')
					time.sleep(1)
			else:
				javab=input("\nDid RASA Recieve it?(y/n)\n")
				if javab=='y':
					break
				else:
					opl=0
		#talker.unadvertise()
		print("\nNow wait for RASA to perform!\n")
		print("\n...")
		input("Are you ready to record?\n\n")
		ttt=time.time()
		print('please stop recording using ctrl+c\n')
		os.system('rostopic echo /RightImit > temp')
		elapsed_time=np.round(time.time()-ttt,2)
		SSS.append(elapsed_time)
		print("\n\nelapsed time = {} seconds.".format(elapsed_time))
		px=norm(return_mat(realtime_path))
		px=px.reshape(1,24,60,1)
		pred=model.predict(px)
		pred_sorted=np.sort(pred)
		ACC.append(pred[0,int(r)])
		print(classes.get(int(give_id(np.round(pred))[0])),' : ',np.round(pred_sorted[0,-1]*100,1),'%') #this line gives the maximum predicted
		if pred_sorted[0,-2] >= 0.005:
			print(classes.get(int(my_search(pred,pred_sorted[0,-2])[0])),' : ',np.round(pred_sorted[0,-2]*100,1),'%')
		print('******************************************')
		if classes.get(int(give_id(np.round(pred))[0])) == word and np.round(pred_sorted[0,-1]*100,1)>=65:
			print('Great! You did it! This word is passed!\n')
			per_rep.append(step+1)
			step=step+1
			word_stat='pass'
			emo=Gen_out_FRB('p',alpha_w,beta_w,gamma_w,alpha_u,beta_u,gamma_u)
			if emo==1:
				emo_id='s'
			elif emo==2:
				emo_id='n'
			else:
				emo_id='h'
			emo_talker.publish(roslibpy.Message({'data': '{}'.format(emo_id)}))
			time.sleep(1)
		else:
			print('Oh no! you should try again!\n')
			emo=Gen_out_FRB('f',alpha_w,beta_w,gamma_w,alpha_u,beta_u,gamma_u)
			if emo==1:
				emo_id='s'
			elif emo==2:
				emo_id='n'
			else:
				emo_id='h'
			emo_talker.publish(roslibpy.Message({'data': '{}'.format(emo_id)}))
			print('sending emotion {}'.format(emo_id))
			time.sleep(1)
			step=step+1
	if len(per_rep)==0:
		print('\nSorry! you failed at this word! we should try another one!\n')
		per_rep.append(out_repeat)
	per_acc=np.average(ACC)
	per_acc=np.round(per_acc,3)
	per_spd=np.average(SSS)
	per_spd=np.round(per_spd,3)
	per_rep=np.min(per_rep)
	write_performance(word_file_a,per_rep,per_acc,per_spd)
	write_performance(user_file_a,per_rep,per_acc,per_spd)
	write_pass_fail(user_path,word_stat,str(int(r)))
	talker.publish(roslibpy.Message({'data': 'STAT = {}'.format(word_stat)}))
	terminator=input("\nContinue Teaching for {}?(y/n)".format(username))
	if terminator=='y':
		print("\nThe Teaching will continue!\n")
	else:
		print("\nThe Teaching is Terminated! Bye!\n")
		finish_talker.publish(roslibpy.Message({'data': 0}))
		break





#######
# OK! THAT'S ALL FOR NOW!
# THANKS FOR VISITING!
# SALAR B
