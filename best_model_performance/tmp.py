import numpy as np
x = np.load('test_ground_truth.npy')
x
x.shape
y = np.load('test_predictions.npy')
y
from sklearn.metrics import average_precision_score
average_precision_score(x,y)
average_precision_score(x[:,0],y[:,0])
average_precision_score(x[:,1],y[:,1])
[average_precision_score(x[:,i],y[:,i]) for i in range(56)]
np.mean([average_precision_score(x[:,i],y[:,i]) for i in range(56)])
%history -f tmp.py
