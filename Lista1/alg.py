import numpy as np
import scipy
import scipy.stats as st

##UTILS
def isNaN(num):
    return num != num

def sgn(num):
  if num > 0:
    return 1
  elif num < 0:
    return -1
  else:
    return 0

def f_polin(x,w):
  return sum([c*np.power(x,i) for i,c in enumerate(w)])

#DISTANCES
def cosine_similarity(a,b):
  return 1 - np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def euclidian (a,b):
  return np.linalg.norm(a-b)

def mahalanobis (a,b):
  c = (a-b)
  return np.sqrt(c.T.dot(np.cov(c)).dot(c)) ## Fazer func cov -.-

#METRICS
def RMSE(x,t,w):
  x = np.c_[ np.ones(x.shape[0]),x ]
  f_x =  x.dot(w.T)  
  return np.sqrt(sum(np.power((t-f_x),2))/x.shape[0])

def RSS(x,t,w):
  x = np.c_[ np.ones(x.shape[0]),x ]
  f_x =  x.dot(w.T)  
  return sum(np.power((t-f_x),2))

def MAPE(x,t,w):
  x = np.c_[ np.ones(x.shape[0]),x ]
  f_x =  x.dot(w.T)  
  return np.sum(np.abs(np.divide(t-f_x,t)))/x.shape[0]

def QME(x,t,w,p):
  x = np.c_[ np.ones(x.shape[0]),x ]
  f_x =  x.dot(w.T)  
  return sum(np.power(t-f_x,2))/(x.shape[0]-p)

def R2 (x,t,w):
  x = np.c_[ np.ones(x.shape[0]),x ]
  f_x =  x.dot(w.T)
  t_bar = np.mean(t)
  num = sum(np.power(t-f_x,2))
  den = sum(np.power(t-t_bar,2))
  return 1-num/den

def accuracy (results, annotation):
  return np.count_nonzero(results == annotation)/ results.shape[0]
  
def recall (results, annotation):
  positives = np.array([results[i] for i,a in enumerate(annotation) if a == 1])
  return np.count_nonzero(positives)/ positives.shape[0]

def precision (results, annotation):
  positives = np.array([annotation[i] for i,r in enumerate(results) if r == 1])
  return np.count_nonzero(positives)/ positives.shape[0]

#HIPOTESIS TESTS
def kendall(points, sig = .05): #Retorna true se ha relacao
  total = 0
  nx = 0
  ny = 0
  n = points.shape[0]
  for i in range(1,n):
    for j in range(0,i):
      x = sgn(points[i,0]-points[j,0])
      y = sgn(points[i,1]-points[j,1])
      if x != 0:
        nx += 1
      if y != 0:
        ny += 1
      total += x*y
  tau = total/(np.sqrt(nx)*np.sqrt(ny))
  
  z = st.norm.ppf(1-sig/2)
  test = z*np.sqrt((4*n+10)/(9*n*n-9*n))
  return abs(tau) > test

def coef_pearson(data):
  x = data[:,0]
  y = data[:,1]
  x_mean = np.mean(x)
  y_mean = np.mean(y)
  n = data.shape[0]
 
  cov_xy = sum([(x[i]-x_mean)*(y[i]-y_mean) for i in range(0,n)])/(n-1)
  var_x = sum([np.power(x_i-x_mean,2) for x_i in x])/(n-1)
  var_y = sum([np.power(y_i-y_mean,2) for y_i in y])/(n-1)
  return cov_xy/(np.sqrt(var_x)*np.sqrt(var_y))

def pearson(data,sig =.05): #Retorna true se ha relacao
  p = coef_pearson(data)
  n = data.shape[0]
  t0 = p*np.sqrt(n-2)/np.sqrt(1-np.power(p,2))
  df = n - 2
  t = st.t.interval(1-sig, df)[1]
  return abs(t0)>t

def snedecor (x,t):
  w = reg_lin(x,t)
  x0 = [np.delete(x, i, axis=1) for i in range(0,x.shape[1])]
  w0 = [reg_lin(x_i,t) for x_i in x0]
  rss = RSS(x,t,w)
  rss0 = [RSS(x0[i],t,w0[i]) for i,x_i in enumerate(x0)]
  
  f = np.array([(rss_i-rss)*(x.shape[0]-x.shape[1]-1)/rss for rss_i in rss0])
  f_1_142 = 3.908# dado pelo exercicio
  
  for i in range (1,w.shape[0]):
    if f[i-1] < f_1_142:
      w[i] = 0 
  
  return w

#ALGORITHMS
def NN_OLD(train_data, train_rotules , val_data):
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
  
def fisher (data, rotules):
  
  mi = np.mean(data, axis = 0)
  
  rotule_names = np.unique(rotules)
  
  data_by_class = [np.array([x for i,x in enumerate(data) if rotules[i]== r])for r in rotule_names]  
  mii = [np.mean(c_data, axis=0) for c_data in data_by_class]
  ni = [c_data.shape[0] for c_data in data_by_class]
  
  si = [ (c_data - mii[i][None,:]).T.dot((c_data - mii[i][None,:])) for i,c_data in enumerate(data_by_class)]
  sw = sum(si)
  
  w = np.linalg.inv(sw).dot(mii[0]-mii[1])

  return data.dot(w)
  
def reg_lin(x,t):
  x = np.c_[ np.ones(x.shape[0]),x ]
  w = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(t)
  return w

def ransac (x,t,s, p=0.99, e=0.2):
  L = np.ceil(np.log(1-p)/np.log(1-np.power(1-e,s)))
  T = x.shape[0]*(1-e)
  tau = np.sqrt(3.84*np.var(t))
  
  data = np.c_[x,t]
  k_best = 0
  while L > 0:
    L-=1
    np.random.shuffle(data)
    train = data[:s,:]
    w = reg_lin(train[:,:-1],train[:,-1])
    f_t = np.c_[np.ones(x.shape[0]),x].dot(w.T)
    k = data[np.abs(f_t-data[:,-1]) < tau ,:]
    w = reg_lin(k[:,:-1],k[:,-1])
    if k.shape[0] > T:
      break
    if k.shape[0] > k_best:
      k_best = k.shape[0]
      best_model = w    
    if L == 0:
      w = best_model
    
  return w 

def rocchio (train_data, train_rotules, data, distance=euclidian):
  rotules = np.unique(train_rotules)
  
  data_by_class = [np.array([x for i,x in enumerate(train_data) if train_rotules[i]== r])for r in rotules]
  centers = [np.mean(c_data,axis = 0) for c_data in data_by_class]
  
  distances = np.array([[distance(d,c) for c in centers] for d in data])
      
  return np.array([rotules[np.argmin(distance)] for distance in distances])

def kNN(train_data, train_rotules , val_data, distance= euclidian,k=1,return_distances=False,distances=None):
  if distances is None:
    distances = np.array([ [distance(t,v) for t in train_data] for v in val_data])

  dist = distances
  min_distances = np.argmin(dist, axis= 1)
  k_result=train_rotules[min_distances]
  k-=1
  dist = np.array([np.delete(elem,min_distances[i]) for i,elem in enumerate(dist)])
  while k > 0:
    k-=1
    min_distances = np.argmin(dist, axis= 1) 
    k_result = np.c_[k_result,train_rotules[min_distances]]
    dist = np.array([np.delete(elem,min_distances[i]) for i,elem in enumerate(dist)])
  
  if (k_result.ndim == 1):
    if return_distances:
      return k_result,distances
    else:
      return k_result
  
  kbin_result = np.apply_along_axis(np.bincount,1,k_result.astype(int),minlength=np.unique(train_rotules).shape[0])  
  if return_distances:
    return np.argmax(kbin_result,axis = 1),distances
  else:
    return np.argmax(kbin_result,axis = 1)
	
def edit (train_data, train_rotules, distance, k = 1,randomize = True):
  data = np.c_[train_data,train_rotules]
  if randomize:
    np.random.shuffle(data)
  edited = data[:k,:]
  for n in range(k,data.shape[0]):
    if (kNN(edited[:,:-1],edited[:,-1],[data[n,:-1]],distance,k = k)[0] != data[n,-1]):
      edited = np.append(edited,[data[n,:]],axis = 0)
  return edited

