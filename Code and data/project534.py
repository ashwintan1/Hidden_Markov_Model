import numpy as np
import time
import matplotlib.pyplot as plt

traindata_orig = np.genfromtxt('train534.dat', dtype = int, delimiter = ' ')
#testdata = np.genfromtxt('test1_534.dat', dtype = int, delimiter = ' ')
traindata = traindata_orig[0:800,:]
cvdata = traindata_orig[800:1000,:]
T = 40
M = 4


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

class model:
    def _init_(self,A,B,pi,ll,niter,llevolution):
        self.A = A
        self.B = B
        self.pi = pi 
        self.ll = ll
        self.niter = niter  
        self.llevolution = llevolution    
          
def baumwelch(N,data):
    A = np.random.rand(N,N)
    A_row_sums = np.sum(A, axis=1)
    A = A/A_row_sums[:,None]
    B = np.random.rand(N,M) 
    B_row_sums = np.sum(B, axis=1)
    B = B/B_row_sums[:,None]
    pi = np.random.rand(1,N)
    pi_sum = np.sum(pi)
    pi = pi/pi_sum
    length = len(data[:,0])
    niter = 0
    l0 = loglikelihood(A,B,pi,data,N)
    llevolution = np.zeros(51)
    l1 = l0
    llevolution[0] = l0
    while ((l1/l0 < 0.9999) or (niter <= 20)) and (niter < 50):
        l0 = l1
        xi_sum = np.zeros((N,N))
        gamma_sum = np.zeros((1,N))
        gamma_one_sum = np.zeros((1,N))
        gamma_pen_sum = np.zeros((1,N))
        B_new_num = np.zeros((N,M))
        for k in range(length):
             x = forwardbackward(data[k,:],A,B,pi,N)
             prob = np.sum(x.alpha[T-1,:])
             gamma =np.multiply(x.alpha,x.beta)/prob
             gamma_one_sum = gamma_one_sum + gamma[0,:]
             gamma_sum = gamma_sum+np.sum(gamma,axis=0)
             gamma_pen_sum = gamma_pen_sum + np.sum(gamma[0:T-1,:],axis=0) 
             for t in range(T):
                 if(t < T-1):
                      for i in range(N):
                          for j in range(N):
                               xi_sum[i,j] = xi_sum[i,j] + x.alpha[t,i]*A[i,j]*x.beta[t+1,j]*B[j,data[k,t+1]]/prob
                 B_new_num[:,data[k,t]] = B_new_num[:,data[k,t]]+np.transpose(gamma[t,:])
        pi = gamma_one_sum/length
        A = xi_sum/np.transpose(gamma_pen_sum)
        B = B_new_num/np.transpose(gamma_sum)
        l1 = loglikelihood(A,B,pi,data,N)
        niter = niter+1
        llevolution[niter] = l1
        
    Model = model()
    Model.A = A
    Model.B = B
    Model.pi = pi
    Model.ll = l1
    Model.niter = niter
    Model.llevolution = llevolution
    return Model            

                                                  
#Cross-Validation

#time1 = time.time()
#niters = np.zeros(6)
#lls = np.zeros(6)
#AIC = np.zeros(6)
#BIC = np.zeros(6)
#for N in range(5,11):
#    Model = baumwelch(N,traindata)
#    niters[N-5] = Model.niter
#    lls[N-5] = loglikelihood(Model.A,Model.B,Model.pi,cvdata,N)
#    AIC[N-5] = -2*lls[N-5] + 2*(N^2+3*N-1)
#   BIC[N-5] = -2*lls[N-5] +np.log(200)*(N^2+3*N-1)
#print lls
#np.savetxt('lls.dat',[lls], delimiter=',')
#print AIC
#np.savetxt('AIC.dat',[AIC], delimiter = ',')
#print BIC
#np.savetxt('BIC.dat',[BIC], delimiter= ',')
#print niters
#time2 = time.time()
#print (time2-time1)

time1 = time.time()
Modelfinal = baumwelch(9, traindata_orig)
pi = Modelfinal.pi
A = Modelfinal.A
B = Modelfinal.B
llevolution = Modelfinal.llevolution
np.savetxt('A0.dat', A, delimiter=',')
np.savetxt('B0.dat', B, delimiter=',')
np.savetxt('pi0.dat', pi, delimiter=',')
np.savetxt('llevolution.dat', llevolution, delimiter = ',')
time2 = time.time()
print (time2-time1)
print llevolution

plt.figure(1)
x = range(0,51)
plt.plot(x,llevolution)
plt.xlabel('Iterations')
plt.ylabel('Log-likelihood')
plt.ylim(bottom= -56000, top = -52000)
plt.savefig('llevolution_image.jpg',color = 'blue', bbox_inches = "tight" )
 

                                  
                      
                                    


       