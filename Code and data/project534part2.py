import numpy as np

A0 = np.genfromtxt('A0.dat', dtype = float, delimiter = ',')
B0 = np.genfromtxt('B0.dat', dtype = float, delimiter = ',')
pi0 = np.genfromtxt('pi0.dat', dtype = float, delimiter = ',')
#print A0
#print A0.max(axis=1)
A0[[0,1,2,3,4,6,7,8]] = A0[[2,3,4,8,7,0,6,1]]
A0[:,[0,1,2,3,4,6,7,8]] = A0[:,[2,3,4,8,7,0,6,1]]
#print A0.max(axis=1)
#print A0.sum(axis=1)
B0[[0,1,2,3,4,6,7,8]] = B0[[2,3,4,8,7,0,6,1]]
pi0[[0,1,2,3,4,6,7,8]] = pi0[[2,3,4,8,7,0,6,1]]
#print pi0
np.savetxt('A.dat', A0, delimiter=',')
np.savetxt('B.dat', B0, delimiter=',')
np.savetxt("pi.dat", [pi0], delimiter=',')