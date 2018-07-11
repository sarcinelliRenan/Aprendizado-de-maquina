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
