# using classes
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import distributions
import torch.nn.functional as F
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
import warnings

def nanmean(tensor, dim = None):
    temp = tensor.detach().clone()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if dim is None:
            return torch.tensor(np.nanmean(temp.numpy()))
        elif dim is not None:
            return torch.tensor(np.nanmean(temp.numpy(), axis = dim))
    

class G(nn.Module):
    def __init__(self, sizes):
        super(G, self).__init__()        
        
        self.location = nn.ModuleList()
        self.location.extend([nn.Sequential(nn.Linear(sizes[i],sizes[i+1]), nn.ReLU()) for i in range(len(sizes)-2)])
        self.location.append(nn.Linear(sizes[-2], sizes[-1]))
        
        self.Scale = nn.ModuleList()
        self.Scale.extend([nn.Sequential(nn.Linear(sizes[i],sizes[i+1]), nn.ReLU()) for i in range(len(sizes)-2)])
        self.Scale.append(nn.Linear(sizes[-2], sizes[-1]))
        
        self.skewness = nn.ModuleList()
        self.skewness.extend([nn.Sequential(nn.Linear(sizes[i],sizes[i+1]), nn.ReLU()) for i in range(len(sizes)-2)])
        self.skewness.append(nn.Linear(sizes[-2], sizes[-1]))          
        
    def loc(self, sequence):
        for layer in self.location:
            sequence = layer(sequence)
        return sequence
    
    def scale(self, sequence):
        for layer in self.Scale:
            sequence = layer(sequence)
        return F.softplus(sequence)
    
    def skew(self, sequence):
        for layer in self.skewness:
            sequence = layer(sequence)
        return sequence

#%%
class MyMethod(object):
    """ Ensemble of deep, heavy tailed, skewed, autoregressive models for time 
    series data.
    """
    
    def __init__(self, dist = ['Normal','Laplace'], device = 
                 'cpu', m = 1000, num = 30, epochs = 300, N = 20, L = [0,1,2], 
                 D = [10,50], lag = [1,10,30,60], tol = [10,30], alphas = 
                 [0.5, 1/3, 0.05, 0.01], split = 0.8, acquisition = 'EI', 
                 iterations = 20, lr = 0.005, prediction_method = 'median'):
        """
        Parameters
        ----------
        dist : iterable with elements in ['Normal','Laplace','t5','Cauchy'],
        default = ['Normal', 'Laplace']
            Conditional distribution of the current observation given the 
            previous observations. The location, scale and skewness of the 
            distribution are modelled by MLPs that take as input the previous 
            'p' observations. The Normal, Laplace, t distribution with 5 
            degrees of freedom (t5) and with 1 degree of freedom (Cauchy) are 
            available options.
            
        device : 'cpu', 'cuda', default = 'cpu'
            Device to train the models on.
            
        m : positive integer, default = 1000
            Number of samples to take from each learner in the ensemble for 
            estimating the quantiles used in the forecasts.
        
        num : positive integer, defaut = 30
            The convergence criteria for training the neural networks is that 
            the previous m evaluations of the loss function have to have a 
            standard deviation less than tol.
            
        epochs : positive integer, default = 300    
            Maximum number of epochs a single learner is allowed to train for 
            if the evaluations of the loss function do not converge.
        
        N : positive integer, default = 20
            Number of learners to use in the ensemble. Each learner is 
            identical up to different random initialisations of the weights 
            and biases of the MLPs before training commences.
        
        L : iterable of non-negative integers, default = [0, 1, 2]
            The number of hidden layers to use in each of the location, scale 
            and skewness MLPs. Bayesian Optimisation is used to find the 
            optimal value of L from the provided options.
        
        D : iterable of non-negative integers, default = [10, 50]
            The width of each hidden in each of the location, scale and 
            skewness MLPs. Bayesian Optimisation is used to find the optimal 
            value of D from the provided options.
            
        lag : iterable with max(lag) < n, default = [1, 10, 30, 60]
            The lag in the autoregressive model. Elements of lag greater than or 
            equal to n are excluded automatically. The observation Y_t depends 
            on observations Y_{t-p} to Y_{t-1}. Bayesian Optimisation is used 
            to find the optimal value of lag from the provided options.
        
        tol : iterable of positive numbers, default = [10, 30]
            The tolerance for convergence during training of the neural 
            networks. Bayesian Optimisation is used to find the optimal 
            tolerance in tol if len(tol) > 1.
        
        alphas : iterable of proportions, default = [0.5, 1/3, 0.05, 0.01]
            The prediction intervals corresponding to the alpha quantiles 
            (alpha probability in tails) are used in Bayesian Optimisation if 
            'uncertainty' is included in the metric in the .fit method. The 
            pinball loss with the validation data is used to assess the fit of 
            the prediciton intervals. Using alpha = 0.5, 1/3 gives 50% and 67% 
            prediction intervals, respectively, assessing if the middle of the 
            conditional distribution is learnt well. Alpha = 0.05, 0.01 gives 
            95% and 99% prediction intervals, useful for assessing if the tails 
            of the conditional distribution are learnt well.
            
        split : float between 0 and 1, default = 0.8
            Proportion of data to use for the training set in a train-
            validation split.
        
        acquisition : 'EI' or 'PI', default = 'EI'
            Acquisition function to be used in Bayesian Optimisation. Choices 
            are between Expected Improvement per second (EI) and Probability of 
            Improvement per second (PI).
            
        iterations : non-negative integer, default = 20
            Number of iterations of Bayesian Optimisation to be run. In theory, 
            more iterations allows for Bayesian Optimisation to find a better 
            set of hyperparameters (smaller validation loss).
            
        lr : float, default = 0.005
            Learning rate for the training of the location, scale and skewness 
            MLPs of each learner. It may be necessary to decrease the learning 
            rate when using the Normal distribution due to unstable training 
            behaviour.
        
        prediction_method : 'median' or 'mean', default = median
            Desired method of making point predictions; use the median 
            (optimised for absolute error) or the mean (optimised for squared 
            error) for predictions. If using the mean, it is recommended to set 
            point_criterion in the loss method to one of 'MSE', 'MSSE' or 
            'RMSSE'.
        """
        super(MyMethod, self).__init__()

        self.split = split
        self.N = N
        self.n = 0
        self.num = num
        self.epochs = epochs
        self.device = torch.device(device)
        self.alphas = alphas
        self.m = m
        self.acquisition = acquisition
        self.best_state_dicts = 0
        self.hyperparams = 0
        self.iterations = iterations
        self.final_loss = 0
        self.lr = lr
        self.prediction_method = prediction_method
        
        self.alphas_new = 0
        self.new_alpha = False
        self.prob = None
        self.increase = None
        
        self.L = L
        self.D = D
        self.p = lag
        self.tol = tol
        self.dist = dist
        
        self.y = 0
#%%
    def fit(self, y, metric = 'accuracy+uncertainty', surrogate = 'GP'):
        """ Fit the model to data.
        
        Parameters
        ----------
        y : torch tensor of size (n, 1)
            Training set of time series data.
        
        metric : 'accuracy', 'uncertainty' or 'accuracy+uncertainty', default 
        ='accuracy+uncertainty'
            The metric to use as the validation loss in Bayesian Optimisation. 
            the accuracy component returns the MASE on the validation data, the 
            uncertainty component returns the sum of the scaled pinball losses
            for all prediction intervals corresponding to alphas. The 
            'accuracy+uncertainty' option uses a scale mixture of both, 
            preferencing models that give accurate point forecasts and 
            prediction intervals. The scaling is 2*accuracy + 1*uncertainty.
        
        surrogate : 'GP' or 'RF', default = GP
            The surrogate model to use for modelling the validation loss as a 
            function of the hyperparameters that are to be learnt. Gaussian 
            Process regression is used if 'GP' is selected, Random Forest 
            regression is used if 'RF' is selected. These methods were chosen 
            due to their ability to give estimates of uncertainty in their 
            predicitons, as well as their favourable performance for small 
            datasets such as the ones arising from Bayesian Optimisation. The
            training time for a given combination of hyperparameters is always 
            modelled by a GP due to the smoothness of the model, this is used 
            for the computation of the per second acquisition function.
        
        Returns
        -------
        self : returns an instance of self.
        """
        self.y = y.clone()
        self.c1_ = y.mean()
        self.c2_ = y.std()
        self.y_scaled = (y.clone() - self.c1_)/self.c2_
        
        self.n = int(self.split*len(y))
        validation = y[self.n:]
        y = y[:self.n]
        
        c1 = y.mean()
        c2 = y.std()
        y = (y-c1)/c2
    
        future = len(validation) # number of time steps into the future to predict
        
        hyperparams = np.zeros([1,5])
        for l in range(len(self.L)):
            for d in range(len(self.D)):
                for lag in range(len(self.p)):
                    for tolerance in range(len(self.tol)):
                        for distribution in range(len(self.dist)):
                            if not ((self.L[l] == 0) and (d > 0.1)):
                                if self.p[lag] < self.n:
                                    hyperparams = np.vstack((hyperparams, np.array([self.L[l], self.D[d], self.p[lag], self.tol[tolerance], self.dist[distribution]])))
        
        hyperparams = hyperparams[1:]
        
        def val_loss(L,D,p,tol,dist,y):
            timer = time.time()
            L = int(L)
            D = int(D)
            p = int(p)
            tol = np.float(tol)
            
            sizes = [p]
            sizes += [D for i in range(L)]
            sizes += [1]
            
            nu5 = False
            if dist == 'Normal':
                rv = distributions.Normal(torch.zeros(1), torch.ones(1)) 
            elif dist == 'Laplace':
                rv = distributions.Laplace(torch.zeros(1), torch.ones(1)) 
            elif dist == 't5':
                nu5 = True # indicate if the t_5 distribution is to be used
                torch.pi = torch.acos(torch.zeros(1)).item() * 2
                sqrt5 = torch.sqrt(torch.Tensor([5])).item()
                c = torch.div(torch.exp(torch.lgamma(torch.Tensor([3]))), torch.mul(torch.sqrt(torch.Tensor([5*torch.pi])), torch.exp(torch.lgamma(torch.Tensor([5/2]))))).item() # t dist constant  
            elif dist == 'Cauchy':
                rv = distributions.Cauchy(torch.zeros(1), torch.ones(1)) 
            
            
            # data preprocessing
            def create_sequences(y, p):
                # input torch tensor of time series data and history length p
                # return list of tuples of input (many) - output (one) tensors
                # n = len(y)
                sequences = []
                for i in range(self.n-p):
                    input_ = y[i:i+p]
                    output_ = y[i+p]
                    sequences += [(input_,output_)]
                    
                return sequences
            
            temp = create_sequences(y, p)
            
            batch = self.n+1 # batch size for training.
            trainLoader = DataLoader(dataset = temp, batch_size = batch, shuffle = False)
            
            # Training loss   
            def t5_cdf(data, loc, scale, skew):
                data = skew*(data-loc)/scale
                return c*(((3*sqrt5*data**4 + 6*sqrt5**3*data**2 + 3*sqrt5**5)*torch.atan(data/sqrt5) + 15*data**3 + 125*data)/(8*data**4 + 80*data**2 + 200) + 3*torch.pi*sqrt5/16)/scale 
             
            def neg_logl(data, loc, scale, skew):
                data = data.squeeze()
                loc = loc.squeeze()
                scale = scale.squeeze()
                skew = skew.squeeze()
                
                if nu5: #if using t_5 distribution, have to define manually
                    rv1 = distributions.StudentT(5,loc, scale)        
                    pdf = torch.exp(rv1.log_prob(data))
                    cdf = t5_cdf(data, loc, scale, skew)
                    temp = 2*torch.mul(pdf,cdf)
                    return -torch.sum(torch.log(temp))
                else:
                    rv.loc = loc.squeeze()
                    rv.scale = scale.squeeze()
                    temp = 2*torch.mul(torch.exp(rv.log_prob(data)), rv.cdf(torch.mul(skew,data)))
                    return -torch.sum(torch.log(temp))
            
            # Training
            self.final_loss = []
            state_dicts = []
            for learner in range(self.N):
                epoch = 0
                losses = [100*(1000+tol)*i for i in range(self.num)]
                g = G(sizes)
                g.to(device=self.device)
                optimiser = torch.optim.Adam(g.parameters(), lr = self.lr)
                y_copy = y.view(-1)
                while ((np.std(losses[-self.num:]) > tol) and (epoch < self.epochs)):
                    for input_, output_ in trainLoader:
                        input_ = input_.view(-1,p).to(device=self.device)
                        output_ = output_.to(device=self.device)
                        optimiser.zero_grad()
                        loc = g.loc(input_)
                        scale = g.scale(input_)
                        skew = g.skew(input_)
                        loss = neg_logl(output_, loc, scale, skew)
                        loss.backward()
                        optimiser.step()
                        losses.append(loss.item())
                    epoch += 1
                #compute loss again after the final update
                loc = g.loc(input_)
                scale = g.scale(input_)
                skew = g.skew(input_)
                loss = neg_logl(output_, loc, scale, skew)
                self.final_loss.append(loss.item())
                state_dicts.append(g.state_dict())
            
                # plt.figure()
                # plt.plot(losses[self.num:])
                # plt.xlabel('iteration')
                # plt.ylabel('loss')
                # plt.show()
            
            # Predictions into future
            def sampler(context, m):
                context = context.to(device = self.device)
                loc = g.loc(context).view(-1)
                scale = g.scale(context).view(-1)
                skew = g.skew(context).view(-1)
                
                if nu5:
                    rv1 = distributions.StudentT(torch.zeros(loc.size()), torch.ones(scale.size()))
                    X = rv1.sample([m])
                    p = t5_cdf(X, torch.zeros(loc.size()), torch.ones(scale.size()), torch.ones(scale.size()))
                else:
                    rv.loc = torch.zeros(loc.size())
                    rv.scale = torch.ones(scale.size())
                    X = rv.sample([m])
                    p = rv.cdf(skew*X)
                
                if not (np.isnan(torch.min(p).item()) or np.isinf(torch.min(p).item())):
                    rv2 = distributions.Bernoulli(p)
                    Z = rv2.sample()
                    X = X.to(device = self.device)
                    Z = Z.to(device = self.device)
                    Y = loc + scale*(2*Z-1)*X
                    return Y.flatten()
                else:
                    return float('nan')*torch.ones(X.flatten().size())
                
              
            point = torch.zeros(future) #point estimate from ensemble
            lower_intervals = torch.zeros(len(self.alphas), future)
            upper_intervals = torch.zeros(len(self.alphas), future)
            
            y_copy = torch.zeros(self.N,self.n)
            for i in range(self.N):
                y_copy[i] = y.flatten()
            
            for i in range(future):
                samples = torch.zeros(self.N, self.m)
                for learner in range(self.N):
                    g = G(sizes)
                    g.to(device=self.device)
                    g.load_state_dict(state_dicts[learner])
                    
                    context = y_copy[learner,-p:].to(device=self.device)
                    samples[learner] = sampler(context, self.m)
                    
                if self.prediction_method == 'median':
                    point[i] = samples.nanquantile(0.5)
                    y_copy = torch.cat((y_copy, samples.nanquantile(0.5, dim=1).view(self.N,1)), dim = 1)
                elif self.prediction_method == 'mean':
                    point[i] = nanmean(samples)
                    y_copy = torch.cat((y_copy, nanmean(samples, dim=1).view(self.N,1)), dim = 1)
                    
                for alpha_index in range(len(self.alphas)):
                    lower_intervals[alpha_index, i] = samples.nanquantile(self.alphas[alpha_index]/2)
                    upper_intervals[alpha_index, i] = samples.nanquantile(1- self.alphas[alpha_index]/2)                
         
            ind_points = y_copy[:,-future:]
            
            # return to orginal data scaling
            y = c2*y + c1
            point = c2*point + c1
            ind_points = c2*ind_points + c1
            
            lower_intervals = c2*lower_intervals + c1
            upper_intervals = c2*upper_intervals + c1
                   
            # Forecast performance
            #point predictions
            if self.prediction_method == 'median':
                ell = torch.abs(validation.flatten() - point.flatten()).mean().item()
            elif self.prediction_method == 'mean':
                ell = torch.sqrt(torch.pow(validation.flatten() - point.flatten(), 2).mean()).item()

            summand = 0
            if self.prediction_method == 'median':
                for i in range(len(y)-1):
                    summand += torch.abs(y[i+1]-y[i])
                summand /= (len(y)-1)
            
            elif self.prediction_method == 'mean':
                for i in range(len(y)-1):
                    summand += (y[i+1]-y[i])**2
                summand /= (len(y)-1)
                summand = torch.sqrt(summand)
                
            summand = summand.item()
            
            # Prediction intervals - pinball loss.
            def pinball(q, y, prob):
                # q: the predicted quantiles
                # y: the test data
                # prob: the probability corresponding to the quantiles
                q = q.flatten()
                y = y.flatten()
                pinballp = (torch.mean((y - q) * prob * (q <= y) + (q - y) * (1-prob) * (q > y))).item()
                scaledp = pinballp/summand
                return scaledp
            
            lower_scaled = torch.zeros(len(self.alphas))
            upper_scaled = torch.zeros(len(self.alphas))
            for alpha_index in range(len(self.alphas)):
                lower_scaled[alpha_index] = pinball(lower_intervals[alpha_index,:], validation, self.alphas[alpha_index]/2)
                upper_scaled[alpha_index] = pinball(upper_intervals[alpha_index,:], validation, 1 - self.alphas[alpha_index]/2)
            
            Loss_accuracy = ell/summand
            Loss_uncertainty = torch.sum(lower_scaled).item() + torch.sum(upper_scaled).item()
            
            runtime = time.time() - timer
            
            #if nan estimates are produced, just swap loss for some large value
            if (torch.tensor(Loss_accuracy).isnan().item() or torch.tensor(Loss_accuracy).isinf().item()):
                Loss_accuracy = 1e50
            if (torch.tensor(Loss_uncertainty).isnan().item() or torch.tensor(Loss_uncertainty).isinf().item()):
                Loss_uncertainty = 1e50
                
            return Loss_accuracy, Loss_uncertainty, runtime, state_dicts
        
        #BO
        if surrogate == 'RF':
            def std(test):
                out = np.zeros([len(test), len(rf.estimators_)])
                for i in range(len(rf.estimators_)):
                    out[:,i] = rf.estimators_[i].predict(test)
                return np.std(out, axis = 1)    
            
            def Z(hyperparams):
                mu = rf.predict(hyperparams).reshape(-1,1)
                sigma = std(hyperparams)
                sigma = sigma.reshape(-1,1)
                return (L_minus - mu)/sigma
            
            if self.acquisition == 'EI':
                def acquisition(hyperparams):
                    # EI
                    z = Z(hyperparams)
                    sigma = std(hyperparams)
                    sigma = sigma.reshape(-1,1)
                    return z*sigma*norm.cdf(z) + sigma*norm.pdf(z)
            
            elif self.acquisition == 'PI':
                def acquisition(hyperparams):
                    # PI
                    z = Z(hyperparams)
                    return norm.cdf(z)
        
        
        elif surrogate == 'GP':
            def Z(hyperparams):
                mu, sigma = gp.predict(hyperparams, return_std = True)
                sigma = 1*sigma.reshape(-1,1)
                return (L_minus - mu)/sigma
            
            if self.acquisition == 'EI':
                def acquisition(hyperparams):
                    # EI
                    z = Z(hyperparams)
                    mu, sigma = gp.predict(hyperparams, return_std = True)
                    sigma = 1*sigma.reshape(-1,1)
                    return z*sigma*norm.cdf(z) + sigma*norm.pdf(z)
            
            elif self.acquisition == 'PI':
                def acquisition(hyperparams):
                    # PI
                    z = Z(hyperparams)
                    return norm.cdf(z)
        
        
        
        indices = np.random.choice(np.arange(len(hyperparams)), size = 1, replace = False)
        H = hyperparams[indices]
        self.hyperparams = H[-1,:]
        
        # def convert_H(H):
        #     # make H usable for surrogate model
        #     temp = np.copy(H)
            
        #     for i in range(H.shape[0]):
        #         if H[i,-1] == 'Normal':
        #             temp[i,-1] = 1
        #         elif H[i,-1] == 'Laplace':
        #             temp[i,-1] = 2
        #         elif H[i,-1] == 't5':
        #             temp[i,-1] = 3
        #         elif H[i,-1] == 'Cauchy':
        #             temp[i,-1] = 4
        #     return temp.astype(np.float32)
        
        def convert_H(H):
            # make H usable for surrogate model
            temp1 = np.zeros([H.shape[0], 4+len(self.dist)])
            for i in range(H.shape[0]):
                temp1[i, :4] = H[i, :4]
                
                for j in range(len(self.dist)):
                    if H[i, -1] == self.dist[j]:
                        temp2 = np.zeros([len(self.dist)])
                        temp2[j] = 1
                        temp1[i, 4:] = temp2
       
            return temp1.astype(np.float32)
        
        val_losses = np.array([[0]])
        duration = np.array([[0]])
        for i in range(H.shape[0]):
            L = H[i,0]
            D = H[i,1]
            p = H[i,2]
            tol = H[i,3]
            dist = H[i,4]
            val_accuracy, val_uncertainty, runtime, parameters = val_loss(L,D,p,tol,dist,y)
            if metric == 'accuracy':
                val_losses = np.vstack((val_losses, np.array([[val_accuracy]])))
            elif metric == 'uncertainty':
                val_losses = np.vstack((val_losses, np.array([[val_uncertainty]])))
            elif metric == 'accuracy+uncertainty':
                val_losses = np.vstack((val_losses, np.array([[2*val_accuracy + val_uncertainty]])))
            
            duration = np.vstack((duration, np.array([[runtime]])))
        
        self.best_state_dicts = parameters
        
        val_losses = val_losses[1:]
        duration = duration[1:]
        
        for iteration in range(self.iterations):
            L_minus = np.min(val_losses)
            
            # print(convert_H(H))
            # print(val_losses)
            
            if surrogate == 'RF':
                rf = RandomForestRegressor(n_estimators = 100, criterion = 'mse')
                rf.fit(convert_H(H), val_losses.reshape(-1))
            
            elif surrogate == 'GP':
                gp = GaussianProcessRegressor(kernel = 1.0*Matern(length_scale = 1.0, length_scale_bounds=(0.01, 10.0), nu=2.5), alpha = 1e-2)
                gp.fit(convert_H(H), val_losses)
            
            gp2 = GaussianProcessRegressor(kernel = 1.0*Matern(length_scale = 1.0, length_scale_bounds=(0.001, 10.0), nu=2.5), alpha = 1e-5)
            gp2.fit(convert_H(H), duration)
            mean_time = gp2.predict(convert_H(hyperparams))
        
            a = acquisition(convert_H(hyperparams)).reshape(-1)/(mean_time.reshape(-1)+0.001)
            H = np.vstack((H,hyperparams[np.argmax(a)]))
            # print(H[-1,:])
            val_accuracy, val_uncertainty, runtime, parameters = val_loss(H[-1,0], H[-1,1], H[-1,2], H[-1,3], H[-1,4], y)
            
            if metric == 'accuracy':
                val_losses = np.vstack((val_losses, np.array([[val_accuracy]])))
            elif metric == 'uncertainty':
                val_losses = np.vstack((val_losses, np.array([[val_uncertainty]])))
            elif metric == 'accuracy+uncertainty':
                val_losses = np.vstack((val_losses, np.array([[2*val_accuracy+val_uncertainty]])))
            duration = np.vstack((duration, np.array([[runtime]])))
            
            if val_losses[-1].item() <= np.min(val_losses):
                self.best_state_dicts = parameters
                self.hyperparams = H[-1,:]
        
        # print(convert_H(H))
        return self
#%%
    def params(self):
        """ Hyperparameters learnt using Bayesian Optimisation 
        
        Returns
        -------
        hyperparams : array of shape (5,)
            Learned hyperparameters of the model of the form: [L,D,p,tol,dist].
        """
        return self.hyperparams

#%%
    def predict(self, future, return_ind = False):
        """ Make a forecast using the trained model
        
        Parameters
        ----------
        future : positive integer
            Number of time steps into the future to make predictions for.
        
        return_ind : bool, default = False
            Indicate whether the point forecasts from each individual learner 
            in the final ensemble should be returned.
        
        Returns
        -------
        point : torch tensor of size (future,)
            Point out-of-sample forecasts.
        
        intervals : tuple of torch tensors of size (len(alphas), future)
            Lower and upper prediciton intervals in order corresponding to the 
            values in alpha.
        
        ind_points : torch tensor of size (N, future)
            Individual point forecasts made by each learner in the final 
            ensemble.
        """
                
        H = self.params()  
        nu5 = False
        if H[-1] == 'Normal':
            rv = distributions.Normal(torch.zeros(1), torch.ones(1)) 
        elif H[-1] == 'Laplace':
            rv = distributions.Laplace(torch.zeros(1), torch.ones(1)) 
        elif H[-1] == 't5':
            nu5 = True # indicate if the t_5 distribution is to be used
            torch.pi = torch.acos(torch.zeros(1)).item() * 2
            sqrt5 = torch.sqrt(torch.Tensor([5])).item()
            c = torch.div(torch.exp(torch.lgamma(torch.Tensor([3]))), torch.mul(torch.sqrt(torch.Tensor([5*torch.pi])), torch.exp(torch.lgamma(torch.Tensor([5/2]))))).item() # t dist constant  
            def t5_cdf(data, loc, scale, skew):
                data = skew*(data-loc)/scale
                return c*(((3*sqrt5*data**4 + 6*sqrt5**3*data**2 + 3*sqrt5**5)*torch.atan(data/sqrt5) + 15*data**3 + 125*data)/(8*data**4 + 80*data**2 + 200) + 3*torch.pi*sqrt5/16)/scale 
        elif H[-1] == 'Cauchy':
            rv = distributions.Cauchy(torch.zeros(1), torch.ones(1))
        
        p = int(H[2])
        L = int(H[0])
        D = int(H[1])
        sizes = [p]
        sizes += [D for i in range(L)]
        sizes += [1]
        
        def sampler(context, m):
            context = context.to(device = self.device)
            loc = g.loc(context).view(-1)
            scale = g.scale(context).view(-1)
            skew = g.skew(context).view(-1)
                   
            if nu5:
                rv1 = distributions.StudentT(torch.zeros(loc.size()), torch.ones(scale.size()))
                X = rv1.sample([m])
                p = t5_cdf(X, torch.zeros(loc.size()), torch.ones(scale.size()), torch.ones(scale.size()))
            else:
                rv.loc = torch.zeros(loc.size())
                rv.scale = torch.ones(scale.size())
                X = rv.sample([m])
                p = rv.cdf(skew*X)
            
            # dont sample anything if something went wrong
            if not (np.isnan(torch.min(p).item()) or np.isinf(torch.min(p).item())):
                rv2 = distributions.Bernoulli(p)
                Z = rv2.sample()
                X = X.to(device = self.device)
                Z = Z.to(device = self.device)
                Y = loc + scale*(2*Z-1)*X
                return Y.flatten()
            else:
                return float('nan')*torch.ones(X.flatten().size())
        
        point = torch.zeros(future) #point estimate from ensemble    
        lower_intervals = torch.zeros(len(self.alphas), future)
        upper_intervals = torch.zeros(len(self.alphas), future)
        
        y_copy = torch.zeros(self.N, len(self.y_scaled))
        for i in range(self.N):
            y_copy[i] = self.y_scaled.flatten()
        
        for i in range(future):
            samples = torch.zeros(self.N, self.m)
            for learner in range(self.N):
                g = G(sizes)
                g.to(device=self.device)
                g.load_state_dict(self.best_state_dicts[learner])
                
                context = y_copy[learner,-p:].to(device = self.device)
                samples[learner] = sampler(context, self.m)
            
            if i == 0:             
                self.prob = nanmean(samples*self.c2_+self.c1_ > self.y[-1]).item()
                self.increase = nanmean(samples*self.c2_+self.c1_ - self.y[-1]).item()
                
            if self.prediction_method == 'median':
                point[i] = samples.nanquantile(0.5)
                y_copy = torch.cat((y_copy, samples.nanquantile(0.5, dim=1).view(self.N,1)), dim = 1)
            elif self.prediction_method == 'mean':
                point[i] = nanmean(samples)
                y_copy = torch.cat((y_copy, nanmean(samples, dim=1).view(self.N,1)), dim = 1)
                    
            for alpha_index in range(len(self.alphas)):
                lower_intervals[alpha_index, i] = samples.nanquantile(self.alphas[alpha_index]/2)
                upper_intervals[alpha_index, i] = samples.nanquantile(1- self.alphas[alpha_index]/2)
             
        # print('samples',samples)
        ind_points = y_copy[:,-future:]
        
        # return to orginal data scaling
        c1 = self.y.mean()
        c2 = self.y.std()
        point = c2*point + c1
        ind_points = c2*ind_points + c1
        
        lower_intervals = c2*lower_intervals + c1
        upper_intervals = c2*upper_intervals + c1
        
        if return_ind:
            return point, (lower_intervals, upper_intervals), ind_points
        else:
            return point, (lower_intervals, upper_intervals)

#%%
    def plot(self, point, intervals, test = None, probs = [0.5,0.67,0.95,0.99], plot_ind = False, ind_points = None, legend = True, save_fig = False):
        """ Plot training data with forecast.
        
        Parameters
        ----------
        point : torch tensor of size (future,)
            Point predictions as outputted by predict method.
            
        intervals : tuple of torch tensors of size (len(alphas), future)
            Tuple containing lower and upper prediction intervals as outputted 
            by predict method.
        
        test : torch tensor of size (n_prime, 1), default = None
            Test data to be plotted along with the forecast. Default is None in
            the case where no test data is available.
        
        probs : iterable containing elements in [0.5, 0.67, 0.95, 0.99], 
        default = [0.5, 0.67, 0.95, 0.99]
            Probabilities corresponding to the coverage of the desired prediciton 
            intervals to be included in the plot.
        
        plot_ind : bool, default = False
            Indicate if the individual point predictions made by each learner 
            in the ensemble should be included in the plot.
            
        ind_points : torch tensor of size (N, future), default = None
            Individual point predictions made by each learner in the final 
            ensemble, as outputted by the predict method.
            
        legend : bool, default = True
            Indicate if a legend should be included in the plot.
            
        save_fig : bool, default = False
            Indicate whether the figure should be saved to file. Filename is
            forecast.pdf.
        
        Returns
        -------
        plot : matplotlib figure
            Training data, point forecasts, test data (if desired) and 
            prediction intervals.
        """
        lower_intervals = intervals[0]
        upper_intervals = intervals[1]
        n = len(self.y)
        future = len(point)
        plt.figure()
           
        colours = ['#4d94ff', '#80b3ff','#b3d1ff','#e6f0ff']
        alpha_indices = []
        
        probs.sort(reverse = True)
        for i in range(len(probs)):
            if probs[i] == 0.99:
                alpha_indices += [3]
            if probs[i] == 0.95:
                alpha_indices += [2]
            if probs[i] == 0.67:
                alpha_indices += [1]
            if probs[i] == 0.5:
                alpha_indices += [0]
        
        counter = 0
        for alpha_index in alpha_indices:
            plt.fill_between(np.linspace(n+1,n+future,future), lower_intervals[alpha_index,:].detach().numpy(), upper_intervals[alpha_index,:].detach().numpy(), color = colours[alpha_index], label = str(int(100*probs[counter]))+'% PI')
            counter += 1
        
        if plot_ind:
            for j in range(self.N-1):
                plt.plot(np.linspace(n+1,n+future,future), ind_points[j].detach().numpy(), c="pink", linewidth=0.8)
            plt.plot(np.linspace(n+1,n+future,future), ind_points[self.N-1].detach().numpy(), c="pink", linewidth=0.8, label='Single Learner Predictions')
            
        plt.plot(np.linspace(1,n,n), self.y.detach().numpy(), c="black", label = 'Training')
        plt.plot(np.linspace(n+1,n+future,future), point.detach().numpy(), c="mediumblue", label = 'Ensemble Prediction', linewidth=2)
        plt.plot(np.linspace(n+1,n+len(test),len(test)), test, c="red", label='Test')
        
        plt.xlabel('$t$')
        plt.ylabel('$y$')
        if legend:
            lgd = plt.legend(markerscale=10, bbox_to_anchor=(1.05, 1), loc='upper left', fancybox=True, shadow=True)
        if save_fig:
            plt.savefig('forecast.pdf', format='pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.show()

        return None

#%%
    def loss(self, test, point, intervals, point_criterion = 'MASE', interval_criterion = 'scaled_pinball'):
        """ Evaluate performance of forecasts.
        
        Parameters
        ----------
        test : torch tensor of size (n_prime, 1)
            Test data, the values that the point forecasts are approximating.
            
        point : torch tensor of size (n_prime,) or (n_prime, 1)
            Point forecasts. Output from predict method accepted.
        
        intervals : tuple of torch tensors of size (len(alpha), n_prime)
            Tuple containing lower and upper prediction intervals.
                
        point_criterion : 'MAE', 'MASE', 'MSE', 'MSSE', 'RMSSE', default='MASE'
            Loss function to use for evaluating the performance of the point 
            forecast. Options include:
            Mean Absolute Error (MAE),
            Mean Absolute Scaled Error (MASE),
            Mean Squared Error (MSE),
            Mean Squared Scaled Error (MSSE),
            Root Mean Squared Scaled Error (RMSSE).
            
        interval_criterion : 'pinball' or 'scaled_pinball', default = 
        'scaled_pinball'
            Loss function for evaluating the performance of the prediction 
            interval forecasts. Scaled Pinball loss is equal to the Pinball 
            Loss divided by the MAE of the predicitons made by the naive random 
            walk sampler on the training data.
        
        Returns
        -------
        L1 : float
            Loss corresponding to the point forecast.
            
        L2 : tuple of floats of size (len(alpha), len(alpha)) 
            Loss corresponding to the prediciton intervals.
        """
        
        lower_intervals = intervals[0]
        upper_intervals = intervals[1]

        # Point predictions
        if ((point_criterion == 'MAE') or (point_criterion == 'MASE')):
            ell = torch.abs(test.flatten() - point.flatten()).mean().item()
            summand = 0
            for i in range(len(self.y)-1):
                summand += torch.abs(self.y[i+1]-self.y[i])
            summand /= (len(self.y)-1)
            summand = summand.item()
        elif ((point_criterion == 'MSE') or (point_criterion == 'MSSE') or (point_criterion == 'RMSSE')):
            ell = torch.pow(test.flatten() - point.flatten(), 2).mean().item()
            summand = 0
            for i in range(len(self.y)-1):
                summand += torch.pow(self.y[i+1] - self.y[i], 2)
            summand /= (len(self.y)-1)
            summand = summand.item()
        
        
        # Prediction intervals
        def pinball(q, y, prob):
            q = q.flatten()
            y = y.flatten()
            pinballp = (torch.mean((y - q) * prob * (q <= y) + (q - y) * (1-prob) * (q > y))).item()
            scaledp = pinballp/summand
            return pinballp, scaledp
            
        lower = torch.zeros(len(self.alphas))
        upper = torch.zeros(len(self.alphas))
        lower_scaled = torch.zeros(len(self.alphas))
        upper_scaled = torch.zeros(len(self.alphas))
        for alpha_index in range(len(self.alphas)):
            lower[alpha_index], lower_scaled[alpha_index] = pinball(lower_intervals[alpha_index,:], test, self.alphas[alpha_index]/2)
            upper[alpha_index], upper_scaled[alpha_index] = pinball(upper_intervals[alpha_index,:], test, 1 - self.alphas[alpha_index]/2)
        
        if point_criterion == 'MAE':
            L1 = ell
        elif point_criterion == 'MASE':
            L1 = ell/summand
        elif point_criterion == 'MSE':
            L1 = ell
        elif point_criterion == 'MSSE':
            L1 = ell/summand
        elif point_criterion == 'RMSSE':
            L1 = torch.sqrt(torch.Tensor([ell/summand])).item()
        
        if interval_criterion == 'pinball':
            L2 = (lower, upper)
        elif interval_criterion == 'scaled_pinball':
            L2 = (lower_scaled, upper_scaled)
        
        return L1, L2

#%%
    def prob_increase(self):
        """ Probability that Y_{n+1} > y_n. Requires self.predict to have been 
        run; uses the results of the simulation for Y_{n+1} to give a Monte 
        Carlo estimate of the probability.
        
        Parameters
        ----------
        self : instance of self
        
        Returns
        -------
        prob : float between 0 and 1
            Probability that Y_{n+1} > y_n.
        """
        
        return self.prob

    def expected_increase(self):
        """ Expected increase; a Monte Carlo estimate of E[Y_{n+1} - y_n] 
        based on the simulation performed in self.predict.
        
        Parameters
        ----------
        self : instance of self
        
        Returns
        -------
        increase : float
            Approximation of E[Y_{n+1} - y_n].
        """
        
        return self.increase
