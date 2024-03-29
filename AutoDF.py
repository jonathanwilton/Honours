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
    # np.nanmean for torch tensors
    temp = tensor.detach().clone()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if dim is None:
            return torch.tensor(np.nanmean(temp.numpy()))
        elif dim is not None:
            return torch.tensor(np.nanmean(temp.numpy(), axis = dim))
    

class G(nn.Module):
    # Neural network class
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
class oracle(object):
    """ Ensemble of deep, heavy tailed, skewed, autoregressive models for time 
    series data.
    """
    
    def __init__(self, dist = ['Normal','Laplace', 't5', 'Cauchy'], m = 500, 
                 num = 30, epochs = 300, N = 30, L = [0,1,2], D = [10,50], 
                 lag = [1,2,5,10,20,50,100,200], tol = [0.5,0.05,5,20], PI = 
                 [0.5,0.67,0.95,0.99], split = 0.8, acquisition = 'EI', 
                 iterations = 10, lr = 0.005, prediction_method = 'median'):
        """
        Parameters
        ----------
        dist : iterable with elements in ['Normal','Laplace','t5','Cauchy'],
        default = ['Normal','Laplace','t5','Cauchy']
            Conditional distribution of the current observation given the 
            previous observations. The location, scale and skewness of the 
            distribution are modelled by MLPs that take as input the previous 
            'lag' observations. The skew Normal, Laplace, t distribution with 5 
            degrees of freedom (t5) and with 1 degree of freedom (Cauchy) are 
            available options.
                        
        m : positive integer, default = 500
            Number of samples to take from each learner in the ensemble for 
            estimating the quantiles used in the forecasts.
        
        num : positive integer, defaut = 30
            Convergence criteria for training the neural networks is the 
            previous 'num' evaluations of the loss function need to have a 
            standard deviation less than 'tol'.
            
        epochs : positive integer, default = 300    
            Maximum number of epochs a single learner is allowed to train for 
            if the evaluations of the loss function do not converge.
        
        N : positive integer, default = 30
            Number of learners to use in the ensemble. Each learner is 
            identical up to different random initialisations of the weights 
            and biases of the MLPs before training commences.
        
        L : iterable of nonnegative integers, default = [0, 1, 2]
            The number of hidden layers to use in each of the location, scale 
            and skewness MLPs. Bayesian Optimisation is used to find the 
            optimal value of L from the provided options.
        
        D : iterable of nonnegative integers, default = [10, 50]
            The width of each hidden layer in each of the location, scale and 
            skewness MLPs. Bayesian Optimisation is used to find the optimal 
            value of D from the provided options.
            
        lag : iterable of nonnegative integers, default = [1,2,5,10,20,50,100,200]
            The lag in the autoregressive model. Elements of lag greater than 
            or equal to n are excluded automatically. The observation Y_t 
            depends on observations Y_{t-lag} to Y_{t-1}. Bayesian Optimisation 
            is used to find the optimal value of lag from the provided options.
        
        tol : iterable of positive numbers, default = [0.5,0.05,5,20]
            The tolerance for convergence during training of the neural 
            networks. Bayesian Optimisation is used to find the optimal 
            tolerance in 'tol'.
        
        PI : iterable of proportions, default = [0.5, 0.67, 0.95, 0.99]
            The coverage probabilities for prediction intervals that are 
            considered in Bayesian Optimisation if 'uncertainty' is included 
            in the metric of the .fit method. The SPL with the validation data 
            is used to assess the fit of the prediction intervals. The 50% and 
            67% prediction intervals are useful for assessing if the middle of 
            the conditional distribution is learnt well. The 95% and 99% 
            prediction intervals are useful for assessing if the tails of the 
            conditional distribution are learnt well.
            
        split : float between 0 and 1, default = 0.8
            Proportion of data to use for the training set in a train/
            validation split.
        
        acquisition : 'EI' or 'PI', default = 'EI'
            Acquisition function to be used in Bayesian Optimisation. Choices 
            are between Expected Improvement per second (EI) and Probability of 
            Improvement per second (PI).
            
        iterations : non-negative integer, default = 10
            Number of iterations of Bayesian Optimisation to be run. In theory, 
            more iterations allows for Bayesian Optimisation to find a better 
            set of hyperparameters (smaller validation loss).
            
        lr : float, default = 0.005
            Learning rate for the training of the location, scale and skewness 
            MLPs of each learner. If there are errors such as no forecast given 
            or the loss is nan, try decreasing the learning rate.
        
        prediction_method : 'median' or 'mean', default = median
            Desired method of making point predictions; use the median 
            (suitable for absolute error metric) or the mean 
            (suitable for squared error metric) for predictions. If using the 
            mean, it is recommended to set point_criterion in the loss method 
            to one of 'MSE', 'MSSE' or 'RMSSE'.
        """
        super(oracle, self).__init__()

        self.split = split
        self.N = N
        self.n = 0
        self.num = num
        self.epochs = epochs
        self.device = torch.device('cpu')
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
        
        self.PI = sorted(PI)
        probs = []
        for probability in PI:
            probs += [(1-probability)/2, (1+probability)/2]
        probs += [0.5]
        self.alphas = sorted(probs)
        
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
            The accuracy component returns the MASE or RMSSE on the validation 
            data, the uncertainty component returns the avergae SPL for all 
            prediction intervals of interest. The 'accuracy+uncertainty' option 
            uses the sum of both, preferencing models that give accurate point 
            forecasts and prediction intervals.
        
        surrogate : 'GP' or 'RF', default = GP
            The surrogate model to use for modelling the validation loss as a 
            function of the hyperparameters that are to be learnt. Gaussian 
            Process regression is used if 'GP' is selected, Random Forest 
            regression is used if 'RF' is selected. These methods were chosen 
            due to their ability to give estimates of uncertainty in their 
            predictions, as well as their favourable performance for small 
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
            tol = float(tol)
            
            sizes = [p]
            sizes += [D for i in range(L)]
            sizes += [1]
            
            nu5 = False
            if dist == 'Normal':
                rv = distributions.Normal(torch.zeros(1), torch.ones(1)) 
            elif dist == 'Laplace':
                rv = distributions.Laplace(torch.zeros(1), torch.ones(1)) 
            elif dist == 't5':
                nu5 = True # indicate the t_5 distribution is to be used
                torch.pi = torch.acos(torch.zeros(1)).item() * 2
                sqrt5 = torch.sqrt(torch.Tensor([5])).item()
                c = torch.div(torch.exp(torch.lgamma(torch.Tensor([3]))), torch.mul(torch.sqrt(torch.Tensor([5*torch.pi])), torch.exp(torch.lgamma(torch.Tensor([5/2]))))).item() # t dist constant  
            elif dist == 'Cauchy':
                rv = distributions.Cauchy(torch.zeros(1), torch.ones(1)) 
            
            
            # data preprocessing
            def create_sequences(y, p):
                # input torch tensor of time series data and history length p
                # return list of tuples of input (many) - output (one) tensors
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
            self.state_dicts = []
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
                self.state_dicts.append(g.state_dict())
                        
            # Predictions into future for validation loss
            def sampler(context, m):
                context = context.to(device = self.device)
                loc = g.loc(context).view(-1)
                scale = g.scale(context).view(-1)
                skew = g.skew(context).view(-1)
                
                if nu5:
                    rv1 = distributions.StudentT(5,torch.zeros(loc.size()), torch.ones(scale.size()))
                    X = rv1.sample([m]).to(device = self.device)
                    p = t5_cdf(X, torch.zeros(loc.size()).to(device = self.device), torch.ones(scale.size()).to(device = self.device), skew)
                else:
                    rv.loc = torch.zeros(loc.size()).to(device = self.device)
                    rv.scale = torch.ones(scale.size()).to(device = self.device)
                    X = rv.sample([m]).to(device = self.device)
                    p = rv.cdf(skew*X)
                
                if not (np.isnan(torch.min(p).item()) or np.isinf(torch.min(p).item())):
                    if ((torch.min(p).item() >= 0) and (torch.max(p).item() <= 1)):
                        rv2 = distributions.Bernoulli(p)
                        Z = rv2.sample()
                        Y = loc + scale*(2*Z-1)*X
                        return Y.flatten()
                    else:
                        return float('nan')*torch.ones(X.flatten().size())
                else:
                    return float('nan')*torch.ones(X.flatten().size())
                
              
            point = torch.zeros(future) #point estimate from ensemble
            intervals = torch.zeros(len(self.alphas), future)
            
            y_copy = torch.zeros(self.N,self.n)
            for i in range(self.N):
                y_copy[i] = y.view(-1)
            
            for i in range(future):
                samples = torch.zeros(self.N, self.m)
                for learner in range(self.N):
                    g = G(sizes)
                    g.to(device=self.device)
                    g.load_state_dict(self.state_dicts[learner])
                    
                    context = y_copy[learner,-p:].to(device=self.device)
                    samples[learner] = sampler(context, self.m)
                    
                if self.prediction_method == 'median':
                    point[i] = samples.nanquantile(0.5)
                    y_copy = torch.cat((y_copy, samples.nanquantile(0.5, dim=1).view(self.N,1)), dim = 1)
                elif self.prediction_method == 'mean':
                    point[i] = nanmean(samples)
                    y_copy = torch.cat((y_copy, nanmean(samples, dim=1).view(self.N,1)), dim = 1)
                    
                for alpha_index in range(len(self.alphas)):
                    intervals[alpha_index, i] = samples.nanquantile(self.alphas[alpha_index])
         
            ind_points = y_copy[:,-future:]
            
            # return to orginal data scaling
            y = c2*y + c1
            point = c2*point + c1
            ind_points = c2*ind_points + c1
            intervals = c2*intervals + c1
                   
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
            
            intervals_scaled = torch.zeros(len(self.alphas))
            for alpha_index in range(len(self.alphas)):
                intervals_scaled[alpha_index] = pinball(intervals[alpha_index,:], validation, self.alphas[alpha_index])
                
            Loss_accuracy = ell/summand
            Loss_uncertainty = torch.mean(intervals_scaled).item()
            
            runtime = time.time() - timer
            
            #if nan estimates are produced, just swap loss for some large value
            if (torch.tensor(Loss_accuracy).isnan().item() or torch.tensor(Loss_accuracy).isinf().item()):
                Loss_accuracy = 1e50
            if (torch.tensor(Loss_uncertainty).isnan().item() or torch.tensor(Loss_uncertainty).isinf().item()):
                Loss_uncertainty = 1e50
                
            return Loss_accuracy, Loss_uncertainty, runtime, self.state_dicts
        
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
                val_losses = np.vstack((val_losses, np.array([[val_accuracy + val_uncertainty]])))
            
            duration = np.vstack((duration, np.array([[runtime]])))
        
        self.best_state_dicts = parameters
        
        val_losses = val_losses[1:]
        duration = duration[1:]
        
        for iteration in range(self.iterations):
            L_minus = np.min(val_losses)
            if surrogate == 'RF':
                rf = RandomForestRegressor(n_estimators = 100, criterion = 'mse')
                rf.fit(convert_H(H), val_losses.reshape(-1))
            
            elif surrogate == 'GP':
                gp = GaussianProcessRegressor(kernel = 1.0*Matern(length_scale = 1.0, length_scale_bounds=(0.0005, 10.0), nu=2.5), alpha = 1e-2)
                gp.fit(convert_H(H), val_losses)
            
            gp2 = GaussianProcessRegressor(kernel = 1.0*Matern(length_scale = 1.0, length_scale_bounds=(0.001, 10.0), nu=2.5), alpha = 1e-5)
            gp2.fit(convert_H(H), duration)
            mean_time = gp2.predict(convert_H(hyperparams))
        
            a = acquisition(convert_H(hyperparams)).reshape(-1)/(mean_time.reshape(-1)+0.001)
            H = np.vstack((H,hyperparams[np.argmax(a)]))
            val_accuracy, val_uncertainty, runtime, parameters = val_loss(H[-1,0], H[-1,1], H[-1,2], H[-1,3], H[-1,4], y)
            
            if metric == 'accuracy':
                val_losses = np.vstack((val_losses, np.array([[val_accuracy]])))
            elif metric == 'uncertainty':
                val_losses = np.vstack((val_losses, np.array([[val_uncertainty]])))
            elif metric == 'accuracy+uncertainty':
                val_losses = np.vstack((val_losses, np.array([[val_accuracy+val_uncertainty]])))
            duration = np.vstack((duration, np.array([[runtime]])))
            
            if val_losses[-1].item() <= np.min(val_losses):
                self.best_state_dicts = parameters
                self.hyperparams = H[-1,:]
        
        return self
#%%
    def params(self):
        """ Hyperparameters learnt using Bayesian Optimisation 
        
        Returns
        -------
        hyperparams : array of shape (5,)
            Learned hyperparameters of the model of the form: [L,D,lag,tol,dist].
        """
        return self.hyperparams

#%%
    def predict(self, future, return_ind = False):
        """ Make a forecast using the trained model
        
        Parameters
        ----------
        future : positive integer
            Number of time steps into the future to make predictions for/
            forecast horizon (h).
        
        return_ind : bool, default = False
            Indicate whether the point forecasts from each individual learner 
            in the final ensemble should be returned.
        
        Returns
        -------
        point : torch tensor of size (future,)
            Point forecasts.
        
        intervals : tuple of torch tensors of size (len(alphas), future)
            Lower and upper prediction intervals in order corresponding to the 
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
            nu5 = True # indicate the t_5 distribution is to be used
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
                rv1 = distributions.StudentT(5, torch.zeros(loc.size()), torch.ones(scale.size()))
                X = rv1.sample([m]).to(device = self.device)
                p = t5_cdf(X, torch.zeros(loc.size()).to(device = self.device), torch.ones(scale.size()).to(device = self.device), skew)
            else:
                rv.loc = torch.zeros(loc.size()).to(device = self.device)
                rv.scale = torch.ones(scale.size()).to(device = self.device)
                X = rv.sample([m]).to(device = self.device)
                p = rv.cdf(skew*X)
            
            # dont sample anything if something went wrong
            if not (np.isnan(torch.min(p).item()) or np.isinf(torch.min(p).item())):
                #also don't sample if something else went wrong
                if ((torch.min(p).item() >= 0) and (torch.max(p).item() <= 1)):
                    rv2 = distributions.Bernoulli(p)
                    Z = rv2.sample()
                    Y = loc + scale*(2*Z-1)*X
                    return Y.flatten()
                else:
                    return float('nan')*torch.ones(X.flatten().size())
            else:
                return float('nan')*torch.ones(X.flatten().size())
        
        point = torch.zeros(future) #point estimate from ensemble    
        intervals = torch.zeros(len(self.alphas), future)
        
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
                intervals[alpha_index, i] = samples.nanquantile(self.alphas[alpha_index])
             
        ind_points = y_copy[:,-future:]
        
        # return to orginal data scaling
        c1 = self.y.mean()
        c2 = self.y.std()
        point = c2*point + c1
        ind_points = c2*ind_points + c1
        self.ind_points = ind_points
        
        intervals = c2*intervals + c1
        
        if return_ind:
            return point, intervals, ind_points
        else:
            return point, intervals

#%%
    def plot(self, point, intervals, test = None, plot_ind = False, 
             ind_points = None, legend = True, save_fig = False):
        """ Plot training data with forecast.
        
        Parameters
        ----------
        point : torch tensor of size (future,)
            Point predictions as outputted by predict method.
            
        intervals : tuple of torch tensors of size (len(alphas), future)
            Tuple containing lower and upper prediction intervals as outputted 
            by predict method.
        
        test : torch tensor of size (h, 1), default = None
            Test data to be plotted along with the forecast. Default is None in
            the case where no test data is available.
                
        plot_ind : bool, default = False
            Indicate if the individual point predictions made by each learner 
            in the ensemble should be included in the plot.
            
        ind_points : torch tensor of size (N, future), default = None
            Individual point predictions made by each learner in the final 
            ensemble, as outputted by the predict method.
            
        legend : bool, default = True
            Indicate if a legend should be included in the plot.
            
        save_fig : bool, default = False
            Indicate whether a copy of the figure should be saved to the 
            directory. Filename is 'forecast.pdf'.
        
        Returns
        -------
        plot : matplotlib figure
            Training data, point forecasts, test data (if desired) and 
            prediction intervals.
        """

        n = len(self.y)
        future = len(point)
        plt.figure()
           
        colours = ['#e6f0ff', '#b3d1ff', '#80b3ff', '#4d94ff']
        
        for index in range(round(len(intervals)/2)):
            plt.fill_between(np.linspace(n+1,n+future,future), intervals[index,:].detach().numpy(), intervals[-(index+1),:].detach().numpy(), color = colours[index], label = str(round(100*self.PI[-(index+1)]))+'% PI')
        
        if plot_ind:
            for j in range(self.N-1):
                plt.plot(np.linspace(n+1,n+future,future), ind_points[j].detach().numpy(), c="pink", linewidth=0.8)
            plt.plot(np.linspace(n+1,n+future,future), ind_points[self.N-1].detach().numpy(), c="pink", linewidth=0.8, label='Single Learner Predictions')
            
        plt.plot(np.linspace(1,n,n), self.y.detach().numpy(), c="black", label = 'Training')
        plt.plot(np.linspace(n+1,n+future,future), point.detach().numpy(), c="mediumblue", label = 'Ensemble Prediction', linewidth=2)
        
        if test is not None:
            plt.plot(np.linspace(n+1,n+len(test),len(test)), test, c="red", label='Test')
        
        plt.xlabel('$t$')
        plt.ylabel('$y$')
        if legend:
            lgd = plt.legend(markerscale=10, bbox_to_anchor=(1.05, 1), loc='upper left', fancybox=True, shadow=True)
        if save_fig:
            if legend:
                plt.savefig('forecast.pdf', format='pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
            else:
                plt.savefig('forecast.pdf', format='pdf', bbox_inches='tight')
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
                
        point_criterion : 'MAE', 'MASE', 'MSE', 'MSSE', 'RMSSE', 'MAPE', 'sMAPE', default='MASE'
            Loss function to use for evaluating the performance of the point 
            forecast. Options include:
            Mean Absolute Error (MAE),
            Mean Absolute Scaled Error (MASE),
            Mean Squared Error (MSE),
            Mean Squared Scaled Error (MSSE),
            Root Mean Squared Scaled Error (RMSSE),
            Mean Absolute Percentage Error (MAPE),
            symmetric Mean Absolute Percentage Error (sMAPE).
            
        interval_criterion : 'pinball', 'scaled_pinball', 'coverage', default = 
        'scaled_pinball'
            Loss function for evaluating the performance of the prediction 
            interval forecasts. Scaled Pinball loss is equal to the Pinball 
            Loss divided by the MAE of the predictions made by the naive 
            one-step look ahead random walk model on the training data. 
            Coverage gives the proportion of points that fall inside the 
            prediction intervals.
        
        Returns
        -------
        L1 : float
            Loss corresponding to the point forecast.
            
        L2 : if interval_criterion == 'pinball' or 'scaled_pinball':
                L2 is a tuple of floats of size (len(alpha), len(alpha)) 
                containing the SPL for all forecasdted quantiles.
            
            else:
                L2 is an array with the first column containing the theoretical 
                prediction interval converage and the second column containing 
                the observed prediction interval coverage. 
                
            
        """
         
        MAE = torch.abs(test.flatten() - point.flatten()).mean().item()
        
        summand = 0
        for i in range(len(self.y)-1):
            summand += torch.abs(self.y[i+1]-self.y[i])
        summand /= (len(self.y)-1)
        abs_scaling = summand.item()
        
        MASE = MAE/abs_scaling
        MSE = torch.pow(test.flatten() - point.flatten(), 2).mean().item()
        
        summand = 0
        for i in range(len(self.y)-1):
            summand += torch.pow(self.y[i+1] - self.y[i], 2)
        summand /= (len(self.y)-1)
        square_scaling = summand.item()
        
        MSSE = MSE/square_scaling
        RMSSE = torch.sqrt(torch.Tensor([MSSE])).item()
        
        temp = torch.abs(test.flatten() - point.flatten())/torch.abs(test.flatten())
        MAPE = temp.mean().item() * 100
        temp = torch.abs(test.flatten() - point.flatten())/(torch.abs(test.flatten())+torch.abs(point.flatten()))
        sMAPE = 2*temp.mean().item() * 100
        
        # Prediction intervals
        def pinball(q, y, prob, scaled):
            q = q.flatten()
            y = y.flatten()
            pinballp = (torch.mean((y - q) * prob * (q <= y) + (q - y) * (1-prob) * (q > y))).item()
            scaledp = pinballp/abs_scaling
            if scaled:
                return scaledp
            else:
                return pinballp
            
        L2 = torch.zeros(len(self.alphas))
        
        if interval_criterion == 'pinball':
            for index in range(len(self.alphas)):
                L2[index] = pinball(intervals[index,:], test, self.alphas[index], False)
        elif interval_criterion == 'scaled_pinball':
            for index in range(len(self.alphas)):
                L2[index] = pinball(intervals[index,:], test, self.alphas[index], True)
        elif interval_criterion == 'coverage':
            L2 = np.zeros([int((len(self.alphas)-1)/2),2])
            for index in range(len(L2)):
                lower_ = intervals[index, :]
                upper_ = intervals[-(index + 1), :]
                proportion = nanmean((lower_ <= test.flatten())*(test.flatten() <= upper_)).item()
                p = 1-2*self.alphas[index] 
                L2[index] = np.array([p, proportion])
            
        
        if point_criterion == 'MAE':
            L1 = MAE
        elif point_criterion == 'MASE':
            L1 = MASE
        elif point_criterion == 'MSE':
            L1 = MSE
        elif point_criterion == 'MSSE':
            L1 = MSSE
        elif point_criterion == 'RMSSE':
            L1 = RMSSE
        elif point_criterion == 'MAPE':
            L1 = MAPE
        elif point_criterion == 'sMAPE':
            L1 = sMAPE
        elif point_criterion == 'all':
            L1 = torch.Tensor([MAPE,sMAPE,MASE,RMSSE])
                
        return L1, L2

#%%
    def prob_increase(self):
        """ Estimate the probability that Y_{n+1} > y_n. Requires 
        self.predict to have been run; uses the results of the simulation for 
        giving the forecast for Y_{n+1} to give a Monte Carlo estimate of the 
        probability.
        
        Parameters
        ----------
        self : instance of self
        
        Returns
        -------
        prob : float between 0 and 1
            Estimate of the probability P(Y_{n+1} > y_n).
        """
        
        return self.prob

    def expected_increase(self):
        """ A Monte Carlo estimate of E[Y_{n+1} - y_n] 
        based on the simulation performed in self.predict.
        
        Parameters
        ----------
        self : instance of self
        
        Returns
        -------
        increase : float
            Estimate of the expectation E[Y_{n+1} - y_n].
        """
        
        return self.increase
