import numpy as np
import scipy

def NN(train_data, train_rotules , val_data):
  dist = np.array([ [np.linalg.norm(t - v) for v in val_data] for t in train_data])
  return train_rotules[np.argmin(dist, axis= 0)]

def pca (data, comp=2, branq=False):
  x = data - np.mean(data, axis=0)[None,:]
  
  C = x.T.dot(x)/(x.shape[0]-1)
  eigen_vals, eigen_vects = np.linalg.eigh(C)
  
  omg_x = x.dot(eigen_vects)
  if (branq):
    V = np.diag(eigen_vals)
    omg_x = omg_x.dot(np.linalg.inv(scipy.linalg.sqrtm(V)).T)

  return omg_x[:, eigen_vals.argsort()[-comp:][::-1]]
