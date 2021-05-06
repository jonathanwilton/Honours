# train on datasets in parallel
import concurrent.futures
import torch
import numpy as np
from AutoDF import oracle
import time
import warnings
import os

M3_train = np.load('M3-train.npy', allow_pickle = True).item()
M3_test = np.load('M3-test.npy', allow_pickle = True).item()
M4_train = np.load('M4-train.npy', allow_pickle = True).item()
M4_test = np.load('M4-test.npy', allow_pickle = True).item()
tourism_train = np.load('tourism-train.npy', allow_pickle = True).item()
tourism_test = np.load('tourism-test.npy', allow_pickle = True).item()

train = [M3_train, tourism_train, M4_train]
test = [M3_test, tourism_test, M4_test]
#train = [M3_train, tourism_train]
#test = [M3_test, tourism_test]

# make list of dataset names
def dict2str(dic):
    out = []
    for dataset_name in dic:
        out += [dataset_name]
    return out

M3_names = dict2str(M3_test)
M4_names = dict2str(M4_test)
tourism_names = dict2str(tourism_test)

all_names = [M3_names, tourism_names, M4_names]
#all_names = [M3_names, tourism_names]

#%%

def L(dataset, ts):
    train_data = torch.Tensor(train[dataset][ts].astype(np.float32)).view(-1,1)
    test_data = torch.Tensor(test[dataset][ts].astype(np.float32)).view(-1,1)
    
    g = oracle(dist = ['Normal', 'Laplace', 't5', 'Cauchy'], device = 'cpu', 
               m = 500, num = 30, epochs = 300, N = 40, L = [0, 1, 2, 3], 
               D = [5, 20, 50, 100], lag = [1, 2, 5, 10, 20, 50, 100, 200], 
               tol = [0.05, 0.5, 5, 20, 50], PI = [0.5, 0.67, 0.95, 0.99], 
               split = 0.8, acquisition = 'EI', iterations = 20, lr = 0.005, 
               prediction_method = 'median')
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g.fit(train_data, metric = 'accuracy+uncertainty', surrogate = 'GP')
    point, intervals = g.predict(future = len(test_data))
    point_loss, interval_loss = g.loss(test_data, point, intervals, point_criterion = 'all', interval_criterion = 'scaled_pinball')
    return point_loss, interval_loss

#%%
def main():
    num_cpus = (
        len(os.sched_getaffinity(0))
        if "sched_getaffinity" in dir(os)
        else os.cpu_count() - 1
    )
    print('num_cpus:', num_cpus)

    point_losses = np.zeros([len(all_names),4])
    interval_losses = np.zeros([len(all_names),9])
    for dataset in range(len(train)):
        timer2 = time.time()
        point_ = np.zeros([len(train[dataset]), 4])
        interval_ = np.zeros([len(train[dataset]), 9])
        
        with concurrent.futures.ProcessPoolExecutor(max_workers = num_cpus) as executor:
            results = executor.map(L, [dataset for i in range(len(all_names[dataset]))], all_names[dataset])
        
        print('dataset '+str(dataset)+' done in',time.time()-timer2,'seconds.')
        
        for series_no, (point_loss, interval_loss) in enumerate(results):
                    point_[series_no] = point_loss
                    interval_[series_no] = interval_loss
        
        point_losses[dataset] = np.nanmean(point_, axis = 0)
        interval_losses[dataset] = np.nanmean(interval_, axis = 0)
        print('\nPoint losses [MAPE, sMAPE, MAsE, RMSSE]:', point_losses)
        print('\nInterval losses [p=0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]:', interval_losses)


if __name__ == '__main__':
    timer = time.time()
    main()
    print('\nTotal time:',time.time()-timer)
