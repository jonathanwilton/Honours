# Nile example - basic
import torch
import pandas as pd
from AutoDF import oracle

nile = pd.read_csv('Nile.csv').values
prop = 0.9 # train-test split proportion
n = int(prop*len(nile))
train = torch.Tensor(nile[:,-1][:n]).view(-1,1)
test = torch.Tensor(nile[:,-1][n:]).view(-1,1)

g = oracle()
g.fit(train)
point, intervals = g.predict(future = len(test))
point_loss, interval_loss = g.loss(test, point, intervals)
g.plot(point, intervals, test = test, save_fig = False)
prob = g.prob_increase()
incr = g.expected_increase()

print('BO learnt [NN depth, NN width, r, tol, dist]:', g.params())
print('MASE:', point_loss)
print('Prediction intervals SPL:') 
for i,p in enumerate([0.005,0.025,0.165,0.25,0.5,0.75,0.835,0.975,0.995]):
    print('SPL('+str(p)+') =', round(interval_loss[i].item(), 4))
print('Probability of increase P(Y_{n+1} > y_n):', prob)
print('Expected increase E[Y_{n+1} - y_n]:', incr)
print('Observed increase y_{n+1} - y_n:', (test[0]-train[-1]).item())