import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, ConstantKernel, WhiteKernel

class GPReg:
    def __init__(self):
        self.kernel = 2**2*RBF(0.5, (1e-1, 1e0)) + WhiteKernel(0.01,(0.09999,0.010001))
        #self.kernel = 4.**2*ExpSineSquared(1.0, 2*np.pi, (1e-5, 1e5), (2*3.11,2*3.15))
    
    def test(self,UX,UY,X):
        UX = np.array(UX)
        UY = np.array(UY)
        X = np.array(X)
        y_pred = np.zeros((X.shape[0],X.shape[1],UY.shape[2]))
        sigma = np.zeros((X.shape[0],X.shape[1],UY.shape[2], UY.shape[2]))
        
        for i in range(X.shape[0]):
            x = X[i,:,:]
            ux = UX[i,:,:]
            uy = UY[i,:,:]
            
            # independent GP for each output dim
            for j in range(UY.shape[2]):
                gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=9)
                if ux.shape[0] > 0:
                    gp.fit(ux,uy[:,j])
            
                y,s = gp.predict(x, return_std=True)
                #y = np.reshape(y,[-1, UY.shape[2]])
                y_pred[i,:,j], sigma[i,:,j,j] = (y,s)
            
        return y_pred, sigma
    
