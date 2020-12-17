#simple 
import torch
import pandas as pd
from classes import MyMethod

data = pd.read_csv('Nile.csv')["Nile"]
y = torch.Tensor(data).view(-1,1)
split = int(0.8*len(y))
train = y[:split]
test = y[split:]
#%%
g = MyMethod()
g.fit(train)
print('hyperparams learnt by BO: [NN depth, hidden layer width, lag, tol, skewed dist] =', g.params())
point, intervals = g.predict(future = len(test))
g.plot(point, intervals, test = test, legend = True)
point_loss, (lower_loss, upper_loss) = g.loss(test, point, intervals)
prob = g.prob_increase()
incr = g.expected_increase()

print('\nPoint predicitions MASE:', point_loss)
print('Lower prediction intervals scaled pinball loss:', lower_loss)
print('Upper prediction intervals scaled pinball loss:', upper_loss)
print('Probability of increase P(Y_{n+1} > y_n):', prob)
print('Expected increase E[Y_{n+1} - y_n]:', incr)
print('Observed increase y_{n+1} - y_n:', (test[0]-train[-1]).item())
