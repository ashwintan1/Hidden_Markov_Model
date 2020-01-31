import numpy as np
import matplotlib.pyplot as plt

lls = np.genfromtxt('lls.dat',dtype = float, delimiter = ',')
AIC = np.genfromtxt('AIC.dat', dtype = float, delimiter = ',')
BIC = np.genfromtxt('BIC.dat', dtype = float, delimiter = ',')
viterbi = np.genfromtxt('viterbi.dat', dtype = int, delimiter = ',')
A = np.genfromtxt('A.dat', dtype = float, delimiter = ',')
B = np.genfromtxt('B.dat', dtype = float, delimiter = ',')

pi = np.genfromtxt('pi.dat', dtype = float, delimiter = ',')
pi = np.resize(pi, (1,9)) 

N = range(5,11)
plt.figure(1)
plt.plot(N, lls)
plt.xlabel('N')
plt.ylabel('Log-likelihood')
plt.savefig('lls_image.jpg', bbox_inches = "tight")

plt.figure(2)
plt.plot(N, AIC)
plt.xlabel('N')
plt.ylabel('AIC')
plt.savefig('AIC_image.jpg', bbox_inches = "tight")

plt.figure(3)
plt.plot(N, BIC)
plt.xlabel('N')
plt.ylabel('BIC')
plt.savefig('BIC_image.jpg', bbox_inches = "tight")

plt.figure(4)
plt.matshow(viterbi3)
plt.xlabel('t')
plt.ylabel('Training data [880:930]')
plt.colorbar()
plt.savefig('viterbi3_image.jpg', bbox_inches = "tight")

plt.figure(5)
plt.imshow(A, cmap='hot', interpolation='nearest')
plt.title('A')
plt.savefig('A_image.jpg', bbox_inches = "tight")

plt.figure(6)
plt.imshow(B, cmap='hot', interpolation='nearest')
plt.title('B')
plt.savefig('B_image.jpg', bbox_inches = "tight")

plt.figure(7)
plt.imshow(pi, cmap='hot', interpolation='nearest')
plt.title('pi')
plt.savefig('pi_image.jpg', bbox_inches = "tight")