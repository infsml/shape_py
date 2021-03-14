import numpy as np
x=np.arange(-5,6,1)

#y=np.piecewise(x,[x<0,x>=0],[lambda k:np.sqrt(-k),lambda k:np.sqrt(k)])
y=((x<0)*np.sqrt(-x))+((x>=0)*np.sqrt(x))