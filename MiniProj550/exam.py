#%%
import numpy as np
#%%
#%%
### Reservoir sampling
##S has items to sample, R will contain the result *)
#%%
np.log(2,2)
#%%
def entropy(v, idx):
    entropies = []
    for i in idx:
        sub_v = v[:,idx]
        total_elements = sub_v.size
        classes = np.unique(sub_v)
        total_entropy = 0
        for c in classes:
            n_elements = sub_v[sub_v == c].size
            p = n_elements/total_elements
            entropy = -p * np.log2(p)
            total_entropy += entropy
        entropies.append(total_entropy)
    return np.array(entropies)

def gain(target_idx, v):
    gains = []
    rows,cols = v.shape
    root_entropy = entropy( v,[target_idx] )
    root_classes = np.unique(v[:,target_idx])
    
    for i in range(cols):
        if i == target_idx:
            continue
        sub_v = v[:,[i,target_idx]]
        sub_classes = np.unique(sub_v[:,0])
        sub_entropies = 0
        for sub_class in sub_classes:
            subset_class = sub_v[sub_v[:,0]==sub_class]
            sub_rows, sub_cols = subset_class.shape
            sub_entropies += sub_rows/rows * entropy(subset_class,[1])[0]
        gain = root_entropy - sub_entropies
        gains.append(gain)
    return np.array(gains)    


mush = np.array([
    [0,0,0,0,0],
    [0,0,1,0,0],
    [1,1,0,1,0],
    [1,0,0,1,1],
    [0,1,1,0,1],
    [0,0,1,1,1],
    [0,0,0,1,1],
    [1,1,0,0,1]
])
root_entropy = entropy(mush,[4])
#entropy(mush,[0,1,2,3])
gains = gain(4,mush)
print(gains)

#%%
def ReservoirSample(s, r):
    n = len(s)
    k = len(r)
    # fill the reservoir array
    for i  in range(k):
        r[i] = s[i]
    ## replace elements with gradually decreasing probability
    #(* randomInteger(a, b) generates a uniform integer from the inclusive range {a, ..., b} *)
    for i in range(k+1,n):
        j = np.random.randint(1,i)
        if j <= k:
            r[j] = s[i]
#%%

(* S has items to sample, R will contain the result *)
def ReservoirSample2(s, r):
    # fill the reservoir array
    for i in range(k):
        r[i] = s[i]
    #(* random() generates a uniform (0,1) random number *)
    W = np.exp(np.log(np.random.randint()/k))

    while i <= n:
      i  = i + np.round(np.log(np.rando())/np.log(1-W)) + 1
      if i <= n:
          #(* replace a random item of the reservoir with item i *)
          [randomInteger(1,k)] := S[i]  // random index between 1 and k, inclusive
          W := W * exp(log(random())/k)

# %%
c = np.array([[1,3]])
c = c/c.sum()
d = -c*np.log(c)
d.sum()
#%%
def entropy(v):
    retval = []
    vt = v.T
    rows = vt.shape[0]
    for i in range(rows):
        c1 =vt[i][vt[i] == 0].size
        c2 = vt[i][vt[i] == 1].size
        p = np.array([c1,c2])
        p = p/(c1+c2)
        p_c2 = c2/vt.size
        ent = -p * np.log(p)
        retval.append(ent.sum())
    return retval

# %%
mush = np.array([
[0	,0	,0	,0	,0]
,[0	,0	,1	,0	,0]
,[1	,1	,0	,1	,0]
,[1	,0	,0	,1	,1]
,[0	,1	,1	,0	,1]
,[0	,0	,1	,1	,1]
,[0	,0	,0	,1	,1]
,[1	,1	,0	,0	,1]
])


# %%
from sklearn import tree
X = mush
y = np.array([[0],[0],[0],[1],[1],[1],[1],[1]]) 
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# %%
tree.plot_tree(clf) 


# %%
X = np.array([
 [1	,1	,1]
,[1	,0	,0]
,[1	,1	,1]
,[1	,1	,0]
,[1	,1	,1]
,[1	,1	,0]
,[0	,1	,1]
,[0	,1	,1]
,[0	,0	,1]
,[0	,0	,0]
])
y = np.array([[1],[0],[1],[1],[0],[0],[0],[1],[1],[1]]) 

import numpy as np
from sklearn.naive_bayes import CategoricalNB
clf = CategoricalNB()
clf.fit(X, y)
CategoricalNB()
c = np.array([[0,1,0]])
print(clf.predict(c))

# %%
