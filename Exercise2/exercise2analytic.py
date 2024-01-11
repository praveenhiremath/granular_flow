from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import sys
import math
from math import *
import scipy
from scipy.integrate import odeint
from sympy import *
from sympy.abc import y


'''
This code was generated to fulfill the requirements for the course Continuum Mechanics FMEN21.
This code is not to be used by others for University submission. However, the code can be used to understand the assignment.

Author: Praveenkumar Hiremath
Email: praveenkumar.hiremath@mek.lth.se (Email at the University)
       praveenkumar.hiremath2911@gmail.com (Private email)
'''

h=0.3
# L, A, B, C and D values are obtained from exercise1 
L=6
A=0.000367538663377
B=-0.657092957247
C=-0.701321988229
D=0.719832986761
theta=pi*30/180
mu=-1
g=9.81

###Velocity calculation
####### Numerical integration
x_dot = symbols('x_dot', cls=Function)
print (x_dot(y))
x_dot = Function("x_dot")(y)
temp=(((A*exp((y/h)*L))+(B*exp(-(y/h)*L))+(C*(y/h))+D))
diffeq = Eq(x_dot.diff(y,y)*mu+temp*g*sin(theta),0)
print ("Differential equation is:")
print (diffeq)


I1=integrate(temp*g*sin(theta))
print ("After integration of above differential equation:", I1)
print ("Using condition at y=0.0, Txy=0.0 (del v/del y=0.0) ")
y=0.0
Integration_constant1=5.73330725377208*y**2 - 3.53078080006271*y - 9.01388571932093e-5*exp(20.0*y) - 0.161152047764827*exp(-20.0*y)#-I1
print ("Integration constant 1 = ",Integration_constant1)

I2=integrate(I1+Integration_constant1)
print ("After integrating the above differential equation twice:", I2)
print ("Using condition at y=0.3, v(y)=0.0 ")
y=0.3
#Integration_constant2=(5.73330725377208*y**2 - 3.53078080006271*y - 9.01388571932093e-5*exp(20.0*y) - 0.161152047764827*exp(-20.0*y))#-I2
Integration_constant2=-(-1.91110241792403*y**3 + 1.76539040003135*y**2 - 0.16124218662202*y + 4.50694285966046e-6*exp(20.0*y) - 0.00805760238824134*exp(-20.0*y))
print ("Integration constant 2 = ",Integration_constant2)

V=(-1.91110241792403*y**3 + 1.76539040003135*y**2 - 0.16124218662202*y + 4.50694285966046e-6*exp(20.0*y) - 0.00805760238824134*exp(-20.0*y))+Integration_constant2

y_coord=np.linspace(0,0.3,50)
yh=np.zeros(shape=(50),dtype=float)
yh=np.divide(y_coord,h)
Vel=np.zeros(shape=(50),dtype=float)
for i in range(0,50,1):
  y=y_coord[i]
  Vel[i]=(-1.91110241792403*y**3 + 1.76539040003135*y**2 - 0.16124218662202*y + 4.50694285966046e-6*exp(20.0*y) - 0.00805760238824134*exp(-20.0*y))+Integration_constant2

max_v=np.amax(abs(-Vel))
print ("Maximum velocity = ",max_v)
Vel=np.divide(Vel,(max_v))

plt.rcParams.update({'font.size': 24})
plt.figure(figsize=(12, 10))

plt.plot(-Vel, yh,'r-+' ,label='L=6, Numerical')#,'r-+'
plt.gca().invert_yaxis()
plt.xlabel('Velocity')
plt.ylabel('y/h')
#plt.title('Numerical Velocity profile')
#plt.savefig('Numerical_Velocity_profile_for_L'+str(6)+'.png')

print ("\nAnalytical solution follows here:")
Int1_fun='(g*sin(theta))*((h*A/L)*exp((y*L)/h)-(h*B/L)*exp(-(y*L)/h)+(C*y^2/2*h)+D*y)+C1'
print ("Integrating the above ODE once gives")
print (Int1_fun)
Int2_fun='(g*sin(theta))*((h*A/L)^2*exp((y*L)/h)+(h*B/L)^2*exp(-(y*L)/h)+(C*y^3/6*h)+(D*y^2)/2)+C1*y+C2'
print ("The analytical solution of the ODE is")
print (Int2_fun)

y=0.0
C1=4.9*((h/6)*(B*exp(-(y/h)*L))-(h/6)*(A*exp((y/h)*L))-(C*y*y/0.6)-D*y)

print ("At y=0, Txy=0. Therefore, C1 = ", C1)

y=0.3
C2=-(-4.9*((h*h/36)*(B*exp(-(y/h)*L))+(h*h/36)*(A*exp((y/h)*L))+(C*y*y/1.8)+(D*y*y)/2)-(C1*y))
print ("At y=0.3, the velocity is zero. Therefore, C2 = ", C2)


y_coord=np.linspace(0,0.3,50)
yh=np.zeros(shape=(50),dtype=float)
yh=np.divide(y_coord,h)
Vel=np.zeros(shape=(50),dtype=float)
for i in range(0,50,1):
  y=y_coord[i]
  Vel[i]=g*sin(theta)*(((h*h)/(L*L))*A*exp((y*L)/h)+((h*h)/(L*L))*B*exp(-(y*L)/h)+((C*y*y*y)/(6*h))+(D*y*y*0.5))+(C1*y)+C2

max_v=np.amax(abs(-Vel))
print ("Maximum velocity (analytical)",max_v)
Vel=np.divide(Vel,(max_v))
plt.plot(-Vel, yh, 'b-o',label='L=6, Analytical') 
plt.gca().invert_yaxis()
plt.gca().invert_yaxis()
plt.xlabel('Velocity')
plt.ylabel('y/h')
plt.title('Analytical and Numerical Velocity profile')
plt.legend(loc='best')
plt.savefig('Analytical_Velocity_profile_for_L'+str(6)+'.png')

print ("Analytical and numerical velocity profiles are compared and saved as: ","Analytical_Velocity_profile_for_L"+str(6)+".png")
