from __future__ import division
import numpy as np
import sys
import math
from math import *


'''
This code was generated to fulfill the requirements for the course Continuum Mechanics FMEN21.
This code is not to be used by others for University submission. However, the code can be used to understand the assignment.

Author: Praveenkumar Hiremath
Email: praveenkumar.hiremath@mek.lth.se (Email at the University)
       praveenkumar.hiremath2911@gmail.com (Private email)
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


# To calculate fitting error
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
