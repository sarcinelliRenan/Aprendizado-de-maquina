def accuracy (x, rotules):
  return np.count_nonzero(x==rotules)/rotules.shape[0]
  
def p_gauss (x, var, mean):
  return np.exp(-((x-mean)**2)/(2*var))/np.sqrt(2*np.pi*var)
  
def H (data,classes):
  class_cnt = [len([x for x in data if x[-1] == c]) for c in classes]
  return sum([-(x/data.shape[0])*math.log(x/data.shape[0],len(classes)) for x in class_cnt if x != 0])

def E (data, A_index, values):
  Avalues = [np.array([x for x in data if x[A_index] == c])for c in values[A_index]]
  return sum([x.shape[0]*H(x,values[-1])/data.shape[0] for x in Avalues])

def create_node_dec (data,i,i_max,values):
  if (data.shape[0]==0):
    return None
  elif i == i_max:
    p_class = [len([x for x in data if x[-1] == c])/data.shape[0] for c in values[-1]]
    return { "Att_index" : values[-1][np.argmax(p_class)], "Nodes" : None}
  else:
    IG = [H(data,values[-1])-E(data,A_i,values) for A_i in range(6)]
    Max_IG = np.argmax(IG)
    nodes = [create_node_dec (np.array([x for x in data if x[Max_IG]==val]),i+1,i_max,values) for val in values[Max_IG]]
    return { "Att_index" : Max_IG , "Nodes" : nodes}

def printTree_dec (dtree,link_names,level) :
  tabs = ""
  for _ in range(level):
    tabs+="   "
  print (tabs+"-> x"+str(dtree["Att_index"]))
  if not dtree["Nodes"] is None:
    for i,subtree in enumerate(dtree["Nodes"]):
      print (tabs+" |")
      print (tabs+link_names[dtree["Att_index"]][i])
      print (tabs+" |")
      printTree_dec(subtree,link_names,level+1)

def dTree_classify (dtree,element, link_names):
  if dtree["Nodes"] is None:
    return dtree["Att_index"]
  else:
    for i,subtree in enumerate(dtree["Nodes"]):
      if link_names[dtree["Att_index"]][i] == element[dtree["Att_index"]]:
        return dTree_classify( subtree,element,link_names)
        break
        
def create_node_reg (data,i,i_max,values):
  if (data.shape[0]==0):
    return None
  elif i == i_max:
    return { "Att_index" : np.mean(data[:,-1].astype(float)), "Nodes" : None}
  else:
    Dsd = np.std(data[:,-1].astype(float))
    attributes_by_values = np.array([np.array([np.array([x for x in data if x[i] == j]) for j in values[i]])for i in range(4)])

    SDR = [Dsd+np.sum([-x.shape[0]/data.shape[0]*np.std(x[:,-1].astype(float)) for x in att if x.shape[0] > 0]) for att in attributes_by_values]
    Max_SDR = np.argmax(SDR)
    nodes = [create_node_reg (np.array([x for x in data if x[Max_SDR]==val]),i+1,i_max,values) for val in values[Max_SDR]]
    return { "Att_index" : Max_SDR , "Nodes" : nodes}

def printTree_reg (dtree,link_names,level) :
  tabs = ""
  for _ in range(level):
    tabs+="   "
  
  if not dtree is None:
    print (tabs+"-> x"+str(dtree["Att_index"]))
    if not dtree["Nodes"] is None:
      for i,subtree in enumerate(dtree["Nodes"]):
        print (tabs+" |")
        print (tabs+link_names[dtree["Att_index"]][i])
        print (tabs+" |")
        printTree_reg(subtree,link_names,level+1)

def rTree_classify (dtree,element, link_names):
  if dtree is None:
    return 0.7
  elif dtree["Nodes"] is None:
      return dtree["Att_index"]
  else:
    for i,subtree in enumerate(dtree["Nodes"]):
      if link_names[dtree["Att_index"]][i] == element[dtree["Att_index"]]:
        return rTree_classify( subtree,element,link_names)
        break

def RMSE(x,t): 
  return np.sqrt(sum(np.power((t-x),2))/x.shape[0])
  
def mesure (rule, D, n_class):
  valid_att = np.count_nonzero(rule[:-1])
  n_c = [x for x in D if np.count_nonzero(rule[:-1] == x[:-1]) == valid_att]
  n_r = len([x for x in n_c if x[-1] == rule[-1]])
  return (n_r+1)/(len(n_c)+n_class)

def create_rule (D,n_class, rule): 
  best = mesure(rule, D, n_class)
  cont = True
  while cont:
    cont = False
    for i in range(len(rule)-1):
      temp_rule = rule.copy()
      temp_rule[i] = '\0'
      val = mesure(temp_rule,D,n_class)
      if val > best:
        best = val
        best_rule = temp_rule
        cont = True
    if cont:
      rule = best_rule
  return rule

def classify(rules,data):
  result_classes = []
  for d in data:
    best_result = None
    for r in rules[:-1]:
      valid_att = np.count_nonzero(r[:-1])
      if np.count_nonzero(r[:-1] == d[:-1]) == valid_att :
        best_result = r[-1]
    if best_result is None:
      best_result = rules[-1][-1]
    result_classes.append(best_result)
  return result_classes
