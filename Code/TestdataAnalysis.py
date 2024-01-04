import numpy as np
import HMMmodule
import time
import matplotlib.pyplot as plt

testdata = np.genfromtxt('../Data_and_Figures/test1_534.dat', dtype = int, delimiter = ' ')
traindata = np.genfromtxt('../Data_and_Figures/train534.dat', dtype = int, delimiter = ' ')
T=40
N=9

pi = np.genfromtxt('../Data_and_Figures/pi.dat',dtype = float, delimiter = ',')
A  = np.genfromtxt('../Data_and_Figures/A.dat',dtype = float, delimiter = ',')
B = np.genfromtxt('../Data_and_Figures/B.dat', dtype = float, delimiter = ',')

#Running Viterbi algorithm to find most likely hidden states in testdata
time_start = time.time()        
viterbi = HMMmodule.Viterbi(pi, A, B, testdata, T, N)
time_end = time.time()
print(time_end-time_start)  
       
np.savetxt('../Data_and_Figures/viterbi.dat', viterbi, fmt='%i', delimiter=',')
plt.figure(1)
plt.matshow(viterbi)
plt.xlabel('t')
plt.ylabel('Sequence No.')
plt.colorbar()
plt.savefig('../Data_and_Figures/viterbi_image.jpg', bbox_inches = "tight")

ll = HMMmodule.loglikelihood(A,B,pi,testdata,N,T)
print(ll)
np.savetxt('../Data_and_Figures/loglik.dat',[ll])

#Running Viterbi algorithm on parts of training data for inspection
trainsample1 = traindata[625:675, :]
trainsample2 = traindata[880:930, :]
viterbi1 = HMMmodule.Viterbi(pi,A, B, trainsample1, T, N)
viterbi2 = HMMmodule.Viterbi(pi,A, B, trainsample2, T, N)
np.savetxt('../Data_and_Figures/viterbi1.dat', viterbi1, fmt='%i', delimiter=',')
np.savetxt('../Data_and_Figures/viterbi2.dat', viterbi2, fmt='%i', delimiter=',')
plt.figure(2)
plt.matshow(viterbi1)
plt.xlabel('t')
plt.ylabel('Training Data [625:675]')
plt.colorbar()
plt.savefig('../Data_and_Figures/viterbi1_image.jpg', bbox_inches = "tight")
plt.figure(3)
plt.matshow(viterbi2)
plt.xlabel('t')
plt.ylabel('Training Data [880:930]')
plt.colorbar()
plt.savefig('../Data_and_Figures/viterbi2_image.jpg', bbox_inches = "tight")

predict = np.zeros((len(testdata),4), dtype = float)
for k in range(len(testdata)):
    x = HMMmodule.forwardbackward(testdata[k,:],A,B,pi,N,T)
    prob = sum(x.alpha[T-1,:])
    gamma = np.multiply(x.alpha[T-1,:],x.beta[T-1,:])/prob
    temp = np.matmul(gamma,A)
    predict[k,:] = np.matmul(temp,B)
    
np.savetxt('../Data_and_Figures/predict.dat',predict,  delimiter=',')

    

