# Nile example - custom
import torch
import pandas as pd
from AutoDF import oracle

nile = pd.read_csv('Nile.csv').values
prop = 0.9 # train-test split proportion
n = int(prop*len(nile))
train = torch.Tensor(nile[:,-1][:n]).view(-1,1)
test = torch.Tensor(nile[:,-1][n:]).view(-1,1)


g = oracle(dist = ['Normal', 'Laplace', 't5', 'Cauchy'], device = 'cpu', 
           m = 500, num = 30, epochs = 200, N = 20, L = [0,1,2], D = [5,20,50], 
           lag = [30,50,70], tol = [0.5,5,20], PI = [0.5, 0.67, 0.95, 0.99], 
           split = 0.8, acquisition = 'EI', iterations = 20, lr = 0.005, 
           prediction_method = 'median')

g.fit(train, metric = 'accuracy+uncertainty', surrogate = 'GP')
point, intervals, ind_points = g.predict(future = len(test), return_ind = True)
g.plot(point, intervals, test = test, plot_ind = True, ind_points = ind_points, legend = True, save_fig = True)
point_loss, interval_loss = g.loss(test, point, intervals, point_criterion = 'all', interval_criterion = 'scaled_pinball')
prob = g.prob_increase()
incr = g.expected_increase()

print('BO learnt [NN depth, NN width, r, tol, dist]:', g.params())
print('MAPE, sMAPE, MASE, RMSSE:', point_loss)
print('Prediction intervals SPL:') 
for i,p in enumerate([0.005,0.025,0.165,0.25,0.5,0.75,0.835,0.975,0.995]):
    print('SPL('+str(p)+') =', round(interval_loss[i].item(), 4))
print('Probability of increase P(Y_{n+1} > y_n):', prob)
print('Expected increase E[Y_{n+1} - y_n]:', incr)
print('Observed increase y_{n+1} - y_n:', (test[0]-train[-1]).item())