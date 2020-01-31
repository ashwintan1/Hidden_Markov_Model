import numpy as np
import time

testdata = np.genfromtxt('test1_534.dat', dtype = int, delimiter = ' ')
#traindata = np.genfromtxt('train534.dat', dtype = int, delimiter = ' ')

#datasamp = traindata[880:930,:]
ltest = len(testdata)
T=40
N=9

pi = np.genfromtxt('pi.dat',dtype = float, delimiter = ',')
A  = np.genfromtxt('A.dat',dtype = float, delimiter = ',')
B = np.genfromtxt('B.dat', dtype = float, delimiter = ',')
class alphabeta:
    def _init_(self,alpha,beta):
        self.alpha = alpha
        self.beta = beta
        
def forwardbackward(O,A,B,pi,N):
        x = alphabeta()
        x.alpha = np.zeros((T,N))
        x.beta = np.zeros((T,N))
        x.alpha[0,:] = np.multiply(pi,np.transpose(B[:,O[0]]))
        x.beta[T-1,:] = np.ones(N)
        for t in range(T-1):
            temp1 = np.matmul(x.alpha[t,:],A)
            x.alpha[t+1,:] = np.multiply(temp1,np.transpose(B[:,O[t+1]]))
            temp2 = np.multiply(x.beta[T-t-1,:],np.transpose(B[:,O[T-t-1]]))
            x.beta[T-t-2,:] = np.matmul(A,temp2)
        return x
        
def loglikelihood(A,B,pi,data,N):
    ll = 0
    for n in range(len(data[:,0])):
        x = forwardbackward(data[n,:],A,B,pi,N)
        prob = sum(x.alpha[T-1,:])
        ll = ll+np.log(prob)
    return ll


time_start = time.time()        
viterbi = np.empty((50,40),dtype=int)
for k in range(ltest):
    delta = np.zeros((T,N), dtype=float)
    psi = np.empty((T,N), dtype=int)
    delta[0,:] = np.multiply(pi,np.transpose(B[:,testdata[k,0]]))
    psi[0,:] = np.matrix([9,9,9,9,9,9,9,9,9])
    for t in range(T-1):
        temp1 = np.transpose(delta[t,:])
        temp2 = np.multiply(A, temp1[:,None])
        temp3 = np.max(temp2,axis=1)
        delta[t+1,:] = np.multiply(temp3, np.transpose(B[:,testdata[k,t+1]]))
        psi[t+1,:] = np.argmax(temp2,axis=1)
    viterbi[k,T-1] = np.argmax(delta[T-1,:])
    for t in range(T-1):
        viterbi[k,T-t-2] = psi[T-t-1,int(viterbi[k,T-t-1])]
time_end = time.time()
print (time_end-time_start)  
       
np.savetxt('viterbi.dat', viterbi, fmt='%i', delimiter=',')

ll = loglikelihood(A,B,pi,testdata,N)
print ll
np.savetxt('loglik.dat',[ll])


predict = np.zeros((50,4), dtype = float)
for k in range(ltest):
    x = forwardbackward(testdata[k,:],A,B,pi,N)
    prob = sum(x.alpha[T-1,:])
    gamma = np.multiply(x.alpha[T-1,:],x.beta[T-1,:])/prob
    temp = np.matmul(gamma,A)
    predict[k,:] = np.matmul(temp,B)
    
np.savetxt('predict.dat',predict,  delimiter=',')

    

