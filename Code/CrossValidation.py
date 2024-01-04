import numpy as np
import HMMmodule 
import time
import matplotlib.pyplot as plt

traindata_orig = np.genfromtxt('../Data_and_Figures/train534.dat', dtype = int, delimiter = ' ')
traindata = traindata_orig[0:800,:]
cvdata = traindata_orig[800:1000,:]
T = 40
M = 4

time1 = time.time()
niters = np.zeros(6)
lls = np.zeros(6)
AIC = np.zeros(6)
BIC = np.zeros(6)
for N in range(5,11):
    Model = HMMmodule.baumwelch(N,M,T,traindata)
    niters[N-5] = Model.niter
    lls[N-5] = HMMmodule.loglikelihood(Model.A,Model.B,Model.pi,cvdata,N,T)
    AIC[N-5] = -2*lls[N-5] + 2*(N^2+3*N-1)
    BIC[N-5] = -2*lls[N-5] +np.log(200)*(N^2+3*N-1)
print(lls)
np.savetxt('../Data_and_Figures/lls.dat',[lls], delimiter=',')
print(AIC)
np.savetxt('../Data_and_Figures/AIC.dat',[AIC], delimiter = ',')
print(BIC)
np.savetxt('../Data_and_Figures/BIC.dat',[BIC], delimiter= ',')
print(niters)
time2 = time.time()
print(time2-time1)

N = range(5,11)
plt.figure(1)
plt.plot(N, lls)
plt.xlabel('N')
plt.ylabel('Log-likelihood')
plt.savefig('../Data_and_Figures/lls_image.png', bbox_inches = "tight")

plt.figure(2)
plt.plot(N, AIC)
plt.xlabel('N')
plt.ylabel('AIC')
plt.savefig('../Data_and_Figures/AIC_image.jpg', bbox_inches = "tight")

plt.figure(3)
plt.plot(N, BIC)
plt.xlabel('N')
plt.ylabel('BIC')
plt.savefig('../Data_and_Figures/BIC_image.jpg', bbox_inches = "tight")