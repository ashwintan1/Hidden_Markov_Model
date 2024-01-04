import numpy as np

class alphabeta:
    def _init_(self,alpha,beta):
        self.alpha = alpha
        self.beta = beta
        
def forwardbackward(O,A,B,pi,N,T):
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

def loglikelihood(A,B,pi,data,N,T):
    ll = 0
    for n in range(len(data[:,0])):
        x = forwardbackward(data[n,:],A,B,pi,N,T)
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
          
def baumwelch(N,M,T,data):
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
    l0 = loglikelihood(A,B,pi,data,N,T)
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
             x = forwardbackward(data[k,:],A,B,pi,N,T)
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
        l1 = loglikelihood(A,B,pi,data,N,T)
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

def Viterbi(pi,A,B,testdata,T,N):
    ltest = len(testdata)
    viterbi = np.empty((ltest,40),dtype=int)
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
    return viterbi