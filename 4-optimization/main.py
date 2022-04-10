# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:08:36 2019
@author: Salar Basiri - 97200346
salarbsr.1996@gmail.com
Genetic Algorithm
HW4 - Intelligent Systems
Dr.Broushaki
"""
#%%
import numpy as np
import sys
from random import choices
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,History
import time

#%%
#sys.stdout=open(r"C:/Users/Salar Basiri/Desktop/log.txt",'w')
x = loadmat(r"C:/Users/Salar Basiri/Desktop/ready_data.mat")
xx=x['out']
data=xx;
target=np.zeros(xx.shape[0])
for i in range(0,xx.shape[0]):
    target[i]=xx[i,0,60]
x_trn=data[:,:,:-1]
y_trn=target    
x_trn,x_vld,y_trn,y_vld=train_test_split(x_trn,y_trn,test_size=0.2,shuffle=True)
x_trn,x_tst,y_trn,y_tst=train_test_split(x_trn,y_trn,test_size=0.2,shuffle=True)
x_trn=x_trn.reshape(x_trn.shape[0],24,60,1)
x_vld=x_vld.reshape(x_vld.shape[0],24,60,1)
x_tst=x_tst.reshape(x_tst.shape[0],24,60,1)
y_trn=to_categorical(y_trn)
y_vld=to_categorical(y_vld)
y_tst=to_categorical(y_tst)

#%%
N_ipop=32;
myu=0.2;
par_num=4;
max_iter=15;
repeat_num=1;

k1_min=30
k1_max=150;
k2_min=30;
k2_max=150;
f1_min=1;
f1_max=6;
f2_min=1;
f2_max=6;





def give_id(x):
    b=np.zeros(x.shape[0])
    for i in range (0,x.shape[0]):
        for j in range(0,x.shape[1]):
            if x[i,j]==1:
                b[i]=j;       
    return b

def un_norm(x):
    x[0]=int(x[0]*(k1_max-k1_min)+k1_min)
    x[1]=int(x[1]*(k2_max-k2_min)+k2_min)
    x[2]=int(x[2]*(f1_max-f1_min)+f1_min)
    x[3]=int(x[3]*(f2_max-f2_min)+f2_min)
    return x
  
  
def return_model(x):
    k1=x[0]
    k2=x[1]
    f1=x[2]
    f2=x[3]
    model=Sequential();
    model.add(Conv2D(k1,kernel_size=f1,activation='relu',input_shape=(24,60,1)))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(k2,kernel_size=f2,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(15,activation='softmax'))
    opt = SGD(lr=10e-5)
    model.compile(optimizer=opt,loss='categorical_crossentropy' , metrics=['accuracy'])
    history=History()
    cback=EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=10,
                                  verbose=0, mode='auto',restore_best_weights=True)
    model.fit(x_trn,y_trn,validation_data=(x_vld,y_vld),epochs=200,shuffle=True,callbacks = [cback,history])
    final_acc=100-(np.count_nonzero(give_id(np.round(model.predict(x_tst)))-give_id(y_tst))/y_tst.shape[0]*100)
    #criteria=-((np.max(history.history['val_acc'])+final_acc/100)/2+0.5*(500-history.epoch[-1])*1e-4)
    criteria=-np.max(history.history['val_acc'])
    return [criteria,final_acc,model]

def mate(x,y):
    u=x.copy()
    v=y.copy()
    r=int(np.random.rand()*(x.shape[1]))
    a=u[0,r]
    b=v[0,r]
    u[0,r]=b
    v[0,r]=a
    uu=u.copy()
    vv=v.copy()
    beta=np.random.rand()
    uu=(1-beta)*u+beta*v
    vv=beta*u+(1-beta)*v
    return uu,vv  



#%% Initial parameters


final_model=Sequential()
dummy=Sequential()
final_acc=0
acc=0
mean_cost=np.zeros(max_iter)
min_cost=np.zeros(max_iter)
for main_count in range(0,repeat_num):   
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Here we gooooo!")
    ipop=np.random.rand(N_ipop,par_num)
    ipop_eval=np.zeros(N_ipop)
    print("Now we evaluate initial population:")
    for i in range (0,N_ipop):
        t=time.clock()
        ipop_eval[i],acc,dummy=return_model(un_norm(ipop[i,:].tolist()))
        print("number ",i+1,"of",N_ipop," is done! Accuracy is: ",acc,"with seq:",un_norm(ipop[i,:].tolist())," time(s):",time.clock()-t)
        
    
    I=np.argsort(ipop_eval)
    npop=np.random.rand(ipop.shape[0],ipop.shape[1])
    for n in range (0,N_ipop):
        k=I[n]
        npop[n,:]=ipop[k,:]
        
    npop=npop[:int(N_ipop/2)]
    N=int(npop.shape[0]/2)
    pop_good=npop[:N]
    
    for coooo in range(1,max_iter+1):
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("repeat:",main_count,"iter:",coooo)
        p=np.array(range(0,N))
        weights=np.zeros(N)
        for i in range(1,N+1):
            weights[i-1]=(N-i+1)/sum(p)
            weights[i-1]=weights[i-1]+1/(15*i)*0.01
            if i<10:
                weights[i-1]=weights[i-1]*4;
        pop_child=pop_good.copy()
        for i in range(0,int(N),2):
            r1=choices(p,weights)   
            r2=choices(p,weights) 
            pop_child[i,:],pop_child[i+1,:]=mate(pop_good[r1,:].reshape(1,par_num),pop_good[r2,:].reshape(1,par_num))
            
        new_pop=np.concatenate((pop_good,pop_child))
        
        myu_num=int(myu*new_pop.shape[0])
        for i in range(1,myu_num+1):
            r1=int(np.random.rand()*(new_pop.shape[0]-1))+1
            r2=int(np.random.rand()*(new_pop.shape[1]))
            new_pop[r1,r2]=np.random.rand()
        print("now we evaluate new population:")           
        eeval=np.zeros(new_pop.shape[0])
        for i in range (0,new_pop.shape[0]):
            t=time.clock()
            eeval[i],acc,dummy=return_model(un_norm(new_pop[i,:].tolist()))
            print("number ",i+1,"of",new_pop.shape[0]," is done! Accuracy is: ",acc,"with seq:",un_norm(new_pop[i,:].tolist())," time(s):",time.clock()-t)
            if (i==0) and (coooo==max_iter):
                print("Time to Report!")
                final_acc=acc
                final_model=dummy
        mean_cost[coooo-1]=np.mean(eeval)
        min_cost[coooo-1]=np.min(eeval)
        I=np.argsort(eeval,kind='stable')
        npop=np.random.rand(new_pop.shape[0],new_pop.shape[1])
        for n in range (0,new_pop.shape[0]):
            k=I[n]
            npop[n,:]=new_pop[k,:]
        new_pop=npop.copy()
        pop_good=new_pop[:N]
    print("process complete!")
    final=pop_good[0,:] 
                   
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")                   
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")                   
print("final params are: " , un_norm(final.tolist()))
print("final accuracy is: ",final_acc)
print("and the final model is:")
final_model.summary()
figure(num=None, figsize=(6, 10), dpi=80, facecolor='w', edgecolor='k')       
plt.subplot(211)
plt.plot(mean_cost,linewidth=2,color='r')
plt.ylabel("Mean cost")
plt.xlabel("Iteration")
ptit="Mean cost diagram"
plt.title(ptit)
plt.subplot(212)
plt.plot(min_cost,linewidth=2,color='b')
plt.ylabel("Min cost")
plt.xlabel("Iteration")
ptit="Min cost diagram"
plt.title(ptit)
plt.show()
#%%
import dill                            #pip install dill --user
filename = 'D:/globalsave.pkl'
dill.dump_session(filename)

# and to load the session again:
#dill.load_session(filename)