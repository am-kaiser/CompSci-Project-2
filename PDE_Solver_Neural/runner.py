import question3_project2_elem as q3p2
import os
import autograd.numpy.random as npr
import autograd.numpy as np
import time

if __name__ == '__main__':
    npr.seed(1)    

    #Experiment list that stores the value of parameters corresponding to index.

    experiments = {}
    experiments[0] = [5, 5, 10, 0.001, [10], False]
    
    experiments[1] = [5, 5, 1000, 0.001, [10], False]
    experiments[2] = [5, 5, 1000, 0.01, [10], False]
    experiments[3] = [5, 5, 1000, 0.1, [10], False]
    
    experiments[4] = [5, 5, 1000, 0.001, [50], False]
    experiments[5] = [5, 5, 1000, 0.01, [50], False]
    experiments[6] = [5, 5, 1000, 0.1, [50], False]
    
    experiments[7] = [5, 5, 1000, 0.001, [100], False]
    experiments[8] = [5, 5, 1000, 0.01, [100], False]
    experiments[9] = [5, 5, 1000, 0.1, [100], False]
    
    experiments[10] = [5, 5, 1000, 0.001, [200], False]
    experiments[11] = [5, 5, 1000, 0.01, [200], False]
    experiments[12] = [5, 5, 1000, 0.1, [200], False]
    
    experiments[17] = [5, 5, 1000, 0.001, [100, 100], False]
    experiments[18] = [5, 5, 1000, 0.01, [100, 100], False]
    experiments[19] = [5, 5, 1000, 0.1, [100, 100], False]
    
    
    experiments[21] = [10, 10, 1000, 0.001, [10], False]
    experiments[22] = [10, 10, 1000, 0.01, [10], False]
    experiments[23] = [10, 10, 1000, 0.1, [10], False]
    
    experiments[24] = [10, 10, 1000, 0.001, [50], False]
    experiments[25] = [10, 10, 1000, 0.01, [50], False]
    experiments[26] = [10, 10, 1000, 0.1, [50], False]
    
    experiments[27] = [10, 10, 1000, 0.001, [100], False]
    experiments[28] = [10, 10, 1000, 0.01, [100], False]
    experiments[29] = [10, 10, 1000, 0.1, [100], False]
    
    experiments[30] = [10, 10, 1000, 0.001, [200], False]
    experiments[31] = [10, 10, 1000, 0.01, [200], False]
    experiments[32] = [10, 10, 1000, 0.1, [200], False]
    
    experiments[37] = [10, 10, 1000, 0.001, [100, 100], False]
    experiments[38] = [10, 10, 1000, 0.01, [100, 100], False]
    experiments[39] = [10, 10, 1000, 0.1, [100, 100], False]
    
    experiments[47] = [15, 15, 1000, 0.001, [100], False]
    experiments[48] = [15, 15, 1000, 0.01, [100], False]
    experiments[49] = [15, 15, 1000, 0.1, [100], False]
    
    
    experiments[101] = [10, 10, 1000, 0.001, [100], True]
    experiments[102] = [10, 10, 1000, 0.01, [100], True]
    experiments[103] = [10, 10, 1000, 0.1, [100], True]
    
    experiments[104] = [5, 5, 1000, 0.001, [100], True]
    experiments[105] = [5, 5, 1000, 0.01, [100], True]
    experiments[106] = [5, 5, 1000, 0.1, [100], True]
    
    experiments[219] = [5, 5, 3000, 0.1, [100, 100], False]
    experiments[239] = [10, 10, 3000, 0.1, [100, 100], False]
    experiments[319] = [5, 5, 5000, 0.1, [100, 100], False]
    experiments[209] = [5, 5, 3000, 0.1, [100], False]
    experiments[309] = [5, 5, 5000, 0.1, [100], False]
    selected_exp = [209, 309]
    for exp in selected_exp:
        #Hyperparameters
        Nx = experiments[exp][0]
        Nt = experiments[exp][1]
        n_iter = experiments[exp][2]
        lmb = experiments[exp][3]
        num_neurons = experiments[exp][4]
        
        SGD = experiments[exp][5]
        
        x = np.linspace(0, 1, Nx)
        t = np.linspace(0, 1, Nt)
        
        t1 = time.time()
        w_and_b, cost_arr = q3p2.solve_pde_nn(x, t, num_neurons, n_iter, lmb, SGD = SGD)
        t2 = time.time()
        print(t2-t1)
        
        #outputs
        g_nn = np.zeros((Nx, Nt))
        g_an = np.zeros((Nx, Nt)) #analytic solution
    
        for ix, x_ in enumerate(x):
            for it, t_ in enumerate(t):
                g_nn[ix, it] = q3p2.g_trial(np.array([x_,t_]), w_and_b)
                g_an[ix, it] = q3p2.g_analytic(np.array([x_,t_]))
    
        #saving the hyper parameters and data
        postfix = "_" + str(exp)
        out_folder = 'output/'
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        
        np.save(out_folder + 'Nx' + postfix + '.npy', Nx)
        np.save(out_folder + 'Nt' + postfix + '.npy', Nt)
        np.save(out_folder + 'n_iter' + postfix + '.npy', n_iter)
        np.save(out_folder + 'lmb' + postfix + '.npy', lmb)
        np.save(out_folder + 'num_neurons' + postfix + '.npy', np.array(num_neurons))
        np.save(out_folder + 'g_nn' + postfix + '.npy', g_nn)
        np.save(out_folder + 'g_an' + postfix + '.npy', g_an)
        np.save(out_folder + 'cost_arr' + postfix + '.npy', cost_arr)
        for layer in range(len(num_neurons)+1):
            np.save(out_folder + 'w_and_b' + postfix + '_layer_' + str(layer+1) + '.npy', w_and_b[layer])
        #w_and_b = np.load('w_and_b.npy', allow_pickle=True)
        np.save(out_folder + 'time'+ postfix + '.npy', t2-t1)
