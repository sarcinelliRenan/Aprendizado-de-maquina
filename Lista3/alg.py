import numpy as np
import time
import math

#EXE 1
def euclidian (a,b):
  return np.linalg.norm(a-b)

def kmeans(k , data, distance=euclidian):
  np.random.shuffle(data)
  centroids = data[:k,:-1]
  while(True):
    clusters = np.array([np.array([x for x in data if np.argmin([euclidian(x[:-1],c)for c in centroids]) == i])for i in range(k)])
    new_centroids = np.array([np.sum(c[:,:-1],axis = 0)/c.shape[0] for c in clusters])
    if (centroids == new_centroids).all():
      break
    else:
      centroids = new_centroids
  return clusters

def hierarch_cluster(d_max,data,distance=euclidian):
  distances = np.array([[i,j,distance(x[:-1],y[:-1])] for i,x in enumerate(data) for j,y in enumerate(data) if i<j])
  clusters = [[d] for d in data]
  for i,_ in enumerate(distances):
    if distances[i][2] <= d_max and distances[i][0] != distances[i][1]:
      clusters[int(distances[i][0])] += clusters[int(distances[i][1])]
      clusters[int(distances[i][1])] = None
      for j in range(i+1,distances.shape[0]):
        if distances[j][0] == distances[i][1]:
          distances[j][0] = distances[i][0]
        if distances[j][1] == distances[i][1]:
          distances[j][1] = distances[i][0]
      
  return np.array([np.array(c) for c in clusters if c is not None])

def cluster_accuracy (clusters):
  n_clusters = clusters.shape[0]
  n_data = n_acc = 0
  for c in clusters:
    n_data += c.shape[0]
    count = np.zeros(n_clusters)
    for x in c:
      count[int(x[-1])-1] += 1
    n_acc += np.max(count)
  return n_acc/n_data
  
#EXE 2
def find_itemsets (data, sup):
  itemsets = []
  n_item = data.shape[1]
  cnt_sup = round(sup*data.shape[0])
  itemset = np.array([[x] for x in range(n_item) if np.count_nonzero(data[:,x]) >= cnt_sup])
  itemsets.append(itemset)
  k = 1
  while k < data.shape[1] :
    itemset = np.array([np.concatenate((a,b)) for a in itemsets[0] for b in itemsets[k-1] if a[0] < b[0]])
    itemset = np.array([x for x in itemset if np.count_nonzero([1 for y in data[:,x] if np.count_nonzero(y)==y.shape[0]]) >=  cnt_sup])
    if itemset.shape[0] == 0:
      break
    itemsets.append(itemset)
    k+=1
  return np.array(itemsets)[1:]

def combine (start,step,step_max,array,combination,comb_list):
  if step > step_max:
    combination = set(combination)
    mask = np.array([(i in combination) for i in range(len(array))])
    comb_list.append((array[mask],array[~mask]))
    return
  for i in range(start,array.shape[0]+step-step_max):
    combine(i+1,step+1,step_max,array,combination+[i],comb_list)

def find_rules(data, itemset, conf):
  rules = []
  for item in itemset:
    for i in range(0,item.shape[0]-1):
      combine(0,0,i,item,[],rules)
  rules = np.array([x for x in rules if np.count_nonzero([1 for i,_ in enumerate(data) if np.count_nonzero(data[i,x[0]])==x[0].shape[0] and #aUb
                                                                                          np.count_nonzero(data[i,x[1]])==x[1].shape[0]])/
                                        np.count_nonzero([1 for i,_ in enumerate(data) if np.count_nonzero(data[i,x[0]])==x[0].shape[0]]) >= conf ])
  for rule in rules:
    print(rule[0]+1,"->",rule[1]+1)
      

def apriori (data, sup, conf):
  itemsets = find_itemsets (data, sup)
  for i in itemsets:
    find_rules(data, i, conf)

#EXE 3    
def perceptron(data, epoch, learning_rate):
  weights = np.random.rand(data.shape[0]-1)
  data = np.c_[ np.ones(data.shape[0]),data ]
  total_error = 1
  while epoch > 0 and total_error > 0:
    epoch -= 1
    total_error = 0
    for d in data:
      xi = d[:-1]
      ci = d[-1]
      phi = 0 if xi.dot(weights) < 0 else 1
      error = ci - phi
      if error != 0:
        weights = weights+learning_rate*xi*error
        total_error += 1
  return weights
  
#EXE 4
def onehot (data):
  classes = np.unique(data)
  return np.array([[1 if x == c else 0 for c in classes] for x in data])
  
def sigmoid(x):
  return 1/(1+np.exp(-x))

def foward (x,w):
  return sigmoid(w.dot(x))

def accuracy (train, labels):
  comp = [ (t == l).all() for t,l in zip(train,labels)]
  return np.count_nonzero(comp)/len(labels)
  
def MLP_train (data, labels, n, max_epochs, learning_rate, max_error):
  n_labels = labels.shape[1]
  
  data = np.c_[np.ones(len(data)),data]# Add bias neuron in input layer
  w0 = np.random.rand(n,data.shape[1])#weights from the input layer to the hidden layer
  w1 = np.random.rand(n_labels,n+1)#weights from the hidden layer to the output layer
  
  error = max_error+1
  epochs = 0
  while epochs < max_epochs and error > max_error:
    error = 0
    epochs += 1
    for x,d in zip(data,labels):
      phi = np.insert(foward(x,w0),0,1)#sample xi foward in hidden layer
      y = foward(phi,w1)#sample xi foward from hidden to output layers
      #print(y-d)
      error += np.sum(np.power(d-y,2))
      delta = np.multiply((y-d),np.multiply(y,1-y))
      #print (delta)
      lmbda = w1.T.dot(delta)
      #print (w1)
      w1 = w1 - learning_rate*np.array([d*phi for d in delta])
      #print (w1)
      der = np.multiply(np.multiply(phi[1:],1-phi[1:]),lmbda[1:])
      w0 = w0 - learning_rate*np.array([d*x for d in der])
    error = error/len(data)
  #print("Stopped after",epochs,"epochs")
  return w0,w1

def MLP_classify(data,w0,w1):
  data = np.c_[np.ones(len(data)),data]
  weighted = np.array([foward(np.insert(foward(x,w0),0,1),w1) for x in data])
  classified = np.zeros(weighted.shape)
  for c,w in zip(classified,weighted):
    c[np.argmax(w)] = 1
  return classified
  
def ELM_train (data, labels, n):
  n_labels = labels.shape[1]
  T = np.array([[1 if i ==1 else -1 for i in x]for x in labels])
  X = np.c_[np.ones(len(data)),data]
  W = np.random.rand(X.shape[1],n)*2-1
  temp_H = X.dot(W)
  H = 1/(1+np.exp(-temp_H))
  beta = np.linalg.pinv(H).dot(T)
  return W,beta

def ELM_classify (data,w,beta):
  X = np.c_[np.ones(len(data)),data]
  temp_H = X.dot(w)
  H = 1/(1+np.exp(-temp_H))
  weighted = H.dot(beta)
  classified = np.zeros(weighted.shape)
  for c,w in zip(classified,weighted):
    c[np.argmax(w)] = 1
  return classified
  
def wilcoxon (data):
  d = np.c_[data[:,0]-data[:,1],data]
  d = np.c_[np.abs(d[:,0]),d]
  d = d[d[:,0].argsort()]
  d = np.c_[d,np.arange(data.shape[0])+1]
  Rp = np.sum([x[-1] for x in d if x[1] > 0])
  Rm = np.sum([x[-1] for x in d if x[1] < 0])
  sig = 81#from the slides 81
  if Rp <= sig:
    print("KNN++ desempenha melhor que KNN com 95% de Confiança")
  elif Rm <= sig:
    print("KNN desempenha melhor que KNN++ com 95% de Confiança")
  else:
    print("Não foi possivel definir qual algoritmo desempenha melhor com 95% de Confiança")
  if Rm >= 18:
    print("KNN++ foi superior estatisticamente que KNN com 95% de Confiança")
  elif Rp >= 18:
    print("KNN foi superior estatisticamente que KNN++ com 95% de Confiança")
  else:
    print("Não foi possivel definir qual algoritmo foi superior estatisticamente com 95% de Confiança") 
