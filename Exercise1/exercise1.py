from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import sys
import math
from math import *
import scipy
import os
from proj_functions import *

'''
This code was generated to fulfill the requirements for the course Continuum Mechanics FMEN21.
This code is not to be used by others for University submission. However, the code can be used to understand the assignment.

Author: Praveenkumar Hiremath
Email: praveenkumar.hiremath@mek.lth.se (Email at the University)
       praveenkumar.hiremath2911@gmail.com (Private email)
'''

curve1=np.loadtxt('solid_line.dat')
curve2=np.loadtxt('dotted_line.dat')
h=0.3
rows=len(curve1)

max_x=np.argmax(curve1[:,0])
print ("Total number of points extracted from the experimental plot of density vs y/h = ",rows)
#print (rows)
print ("Maximum density occurs at (x,y) = ","(",curve1[max_x,0],",",curve1[max_x,1],")")
#print ("(",curve1[max_x,0],",",curve1[max_x,1],")")

'''
def functions_bcs(L,curve1,max_x,h,rows):

  h=1
  ymax_index=np.argmax(curve1[:,1])
  A=np.zeros(shape=(4,4),dtype=float)
  a11=exp((curve1[0,1])*L)
  a12=exp(-(curve1[0,1])*L)
  a13=curve1[0,1]
  a14=1.0

  a21=exp((curve1[max_x,1])*L)
  a22=exp(-(curve1[max_x,1])*L)
  a23=curve1[max_x,1]
  a24=1.0

  a31=exp((curve1[ymax_index,1])*L)
  a32=exp(-(curve1[ymax_index,1])*L)
  a33=curve1[ymax_index,1]
  a34=1.0
  
  a41=(L/h)*(exp((curve1[max_x,1])*L))
  a42=(-L/h)*(exp(-(curve1[max_x,1])*L))
  a43=1/h
  a44=0.0

  Matrix_B=np.array([curve1[0,0],curve1[max_x,0],curve1[ymax_index,0],0])
  Matrix_A=np.array([[a11,a12,a13,a14],[a21,a22,a23,a24],[a31,a32,a33,a34],[a41,a42,a43,a44]])
  return Matrix_A,Matrix_B
'''

A=np.zeros(20)
B=np.zeros(20)
C=np.zeros(20)
D=np.zeros(20)
for i in range(0,20,1):
  L=i+1
  Matrix_A,Matrix_B=functions_bcs(L,curve1,max_x,h,rows)
  A[i],B[i],C[i],D[i]=(np.linalg.solve(Matrix_A,Matrix_B))

'''
def calc_error(A,B,C,D,curve1,rows):
  Difference2=np.zeros(shape=(20,rows),dtype=float)
  error=np.zeros(shape=(20),dtype=float)
  for i in range(0,20,1):
   vy=np.zeros(shape=(rows),dtype=float)
   L=i+1
   for j in range(0,rows,1):
    
    vy[j]=(A[i]*exp(curve1[j,1]*L))+(B[i]*(exp(-curve1[j,1]*L)))+(C[i]*curve1[j,1])+D[i]
    temp=np.subtract(vy,curve1[:,0])
   Difference2[i,:]=temp
   vy_mean=np.mean(vy)
   Difference1=(curve1[:,0]-vy_mean)
   sigma_square=np.sum(np.square(Difference1))/(rows-1)

  return Difference2 
'''    

Difference2=calc_error(A,B,C,D,curve1,rows)

print ("Optimal values are: A = ",A[5],",","B = ",B[5],",","C = ",C[5],",","D = ",D[5],", L = 6")


for i in range(0,20,1):
  error=np.zeros(shape=(20),dtype=float)
  error[i]=Difference2[i,0]
#print error  
index=np.argmin(Difference2[:,0])
Opt_L=index+1
Opt_A=A[index]
Opt_B=B[index]
Opt_C=C[index]
Opt_D=D[index]
   

path = './exercise1_outputs/'

# Check whether the specified path exists or not
exists = os.path.exists(path)

if not exists:
  
  # Create a new directory because it does not exist 
  os.makedirs(path)
  print("The new directory is created!")


for i in range(0,20,1): 
   vy=np.zeros(shape=(rows,20),dtype=float)
   new=np.zeros(shape=(rows,2),dtype=float)
   L=i+1
   for j in range(0,rows,1):
     vy[j,i]=(A[i]*exp(curve1[j,1]*L))+(B[i]*(exp(-curve1[j,1]*L)))+(C[i]*curve1[j,1])+D[i]
   new[:,0]=vy[:,i]
   new[:,1]=curve1[:,1]
   np.savetxt('./exercise1_outputs/ABCD_for_L'+str(int(i+1))+'.dat',new)


'''
Plotting Solids fraction ($\\nu$) vs y/h 
'''
print ("Plotting Solids fraction ($\\nu$) vs y/h (Experimental and fitted data)...")
plt.rcParams.update({'font.size': 24})
plt.figure(figsize=(12, 10))
plt.plot(curve1[:,1],curve1[:,0],  'ro',label='Expt. data')#'ro',

calc_data=np.loadtxt('./exercise1_outputs/ABCD_for_L6.dat')
plt.plot(calc_data[:,1],calc_data[:,0],'go', label='L=6, FIT') 

plt.ylim(0,0.8)
plt.xlim(0,1.0)
plt.xlabel('y/h')
plt.ylabel('Solids fraction ($\\nu$)')
plt.title('Solids fraction ($\\nu$) vs y/h')
plt.legend(loc='best')
plt.savefig('ABCDL_Vol_frac_fit_L'+str(6)+'.png')

print ("Plotting finished and is saved as: ","ABCDL_Vol_frac_fit_L"+str(6)+".png")

