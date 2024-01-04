import numpy as np
import HMMmodule
import time
import matplotlib.pyplot as plt

traindata_orig = np.genfromtxt('../Data_and_Figures/train534.dat', dtype = int, delimiter = ' ')
T = 40
M = 4
N = 9

time1 = time.time()
Modelfinal = HMMmodule.baumwelch(N,M,T,traindata_orig)
time2 = time.time()

pi = Modelfinal.pi
A = Modelfinal.A
B = Modelfinal.B
llevolution = Modelfinal.llevolution
print(time2-time1)
print(A)
print(B)
print(pi)
print(llevolution)
np.savetxt('../Data_and_Figures/A0.dat', A, delimiter=',')
np.savetxt('../Data_and_Figures/B0.dat', B, delimiter=',')
np.savetxt('../Data_and_Figures/pi0.dat', pi, delimiter=',')
np.savetxt('../Data_and_Figures/llevolution.dat', llevolution, delimiter = ',')

plt.figure(1)
x = range(0,51)
plt.plot(x,llevolution)
plt.xlabel('Iterations')
plt.ylabel('Log-likelihood')
plt.ylim(bottom= -56000, top = -52000)
plt.savefig('../Data_and_Figures/llevolution_image.jpg', bbox_inches = "tight" )

#Relabel hidden states such that states with more unifrom emission probabilities have a lower index.
A[[0,1,2,3,4,6,7,8]] = A[[2,3,4,8,7,0,6,1]]
A[:,[0,1,2,3,4,6,7,8]] = A[:,[2,3,4,8,7,0,6,1]]
B[[0,1,2,3,4,6,7,8]] = B[[2,3,4,8,7,0,6,1]]
pi[:,[0,1,2,3,4,6,7,8]] = pi[:,[2,3,4,8,7,0,6,1]]
np.savetxt('../Data_and_Figures/A.dat', A, delimiter=',')
np.savetxt('../Data_and_Figures/B.dat', B, delimiter=',')
np.savetxt("../Data_and_Figures/pi.dat", pi, delimiter=',')

plt.figure(2)
plt.imshow(A, cmap='hot', interpolation='nearest')
plt.title('A')
plt.savefig('../Data_and_Figures/A_image.jpg', bbox_inches = "tight")

plt.figure(3)
plt.imshow(B, cmap='hot', interpolation='nearest')
plt.title('B')
plt.savefig('../Data_and_Figures/B_image.jpg', bbox_inches = "tight")

plt.figure(4)
plt.imshow(pi, cmap='hot', interpolation='nearest')
plt.title('pi')
plt.savefig('../Data_and_Figures/pi_image.jpg', bbox_inches = "tight")