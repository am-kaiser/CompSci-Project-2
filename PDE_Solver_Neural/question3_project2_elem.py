import autograd.numpy as np
from autograd import jacobian,hessian,grad
import autograd.numpy.random as npr
import time

def sigmoid(x):
    #Signmoid Activation Function
    return 1/(1+np.exp(-x))

def ReLU(x):
    #ReLu Activation Function
    return np.max(0, x)

def ffnn2d(w_and_b, x, t):
    # Feed forwards input through a fully connected neural network.
    # The length of list w_and_b is 1 greater than number of hidden layers.
    # The ith index corresponds to i+1th hidden layer for i < len(w_and_b)
    # The last index stores the value of weights of the output layer.

    N_hidden = np.size(w_and_b) - 1
    
    x_old = x.T
    t_old = t.T
    
    bias = np.ones(1)
    
    #x_and_t_old = np.column_stack((bias, x_old, t_old))
    x_and_t_old = np.row_stack((bias, x_old, t_old))
    #print("x shape is ", x_old.shape)
    #print("input for hidden 1 shape is ", x_and_t_old.shape)
    
    for i in range(N_hidden):
        w_and_b_hidden = np.array(w_and_b[i])
        #print("shape of weights and biases for layer ", i+1, " is ", w_and_b_hidden.shape)
        z_hidden = np.matmul(w_and_b_hidden, x_and_t_old)
        x_and_t_old = sigmoid(z_hidden)
        #print("shape of output of layer ", i+1, " is ", x_and_t_old.shape)
        
        bias = np.ones(x_and_t_old.shape[1])
        #print(bias.shape)
        x_and_t_old = np.row_stack((bias, x_and_t_old))
        #print("shape of input of layer ", i+2, " is ", x_and_t_old.shape)
        
    w_and_b_out = w_and_b[-1]
    output = np.matmul(w_and_b_out, x_and_t_old)
    
    return output[0][0]

def g_trial(xt, w_and_b):
    #trial function that satisfies boundary and initial conditions
    x,t = xt
    return np.sin(np.pi*x) + t*x*(1-x)*ffnn2d(w_and_b, x, t)

def g_analytic(xt):
    x,t = xt
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)

def cost(w_and_b, x, t):
    g_t_jacobian_func = jacobian(g_trial)
    g_t_hessian_func = hessian(g_trial)
    
    cost = 0
    count = 0
    for x_ in x:
        for t_ in t:
            xt = np.array([x_,t_])
            #g_t = g_trial(xt, w_and_b)
            g_t_jacobian = g_t_jacobian_func(xt, w_and_b)
            g_t_hessian = g_t_hessian_func(xt, w_and_b)
    
            dg_t_by_dt = g_t_jacobian[1]
            d2g_t_by_dx2 = g_t_hessian[0, 0]
    
            cost += ((dg_t_by_dt - d2g_t_by_dx2)**2)
            count += 1
    cost /= count
    return cost

def solve_pde_nn(x, t, num_neurons, n_iter, lmb, SGD = False, SGD_ratio = 0.8):
    """
    Parameters:
    x: position array
    t: time array
    num_neurons: list of integers where each index corresponds to number of neurons in the subsequent hidden layer
    n_iter: number of iterations
    lmb: learning rate
    SGD: boolean, if True, SGD performed
    SGD_ratio: fraction of data randomly selected for training the neural network in each iteration

    Output:
    w_and_b: trained weights and biases
    cost_arr: cost after each iteration
    """
    N_hidden = np.size(num_neurons)
    w_and_b = [None]*(N_hidden + 1)
    w_and_b[0] = npr.randn(num_neurons[0], 2 + 1 ) # 2 since we have two points, +1 to include bias
    for l in range(1,N_hidden):
        w_and_b[l] = npr.randn(num_neurons[l], num_neurons[l-1] + 1) # +1 to include bias
    # For the output layer
    w_and_b[-1] = npr.randn(1, num_neurons[-1] + 1 ) # +1 since bias is included

    cost_grad_func = grad(cost, 0)
    cost_arr = np.zeros(n_iter)
    
    if(SGD):
        for iteration in range(n_iter):
            cost_grad = cost_grad_func(w_and_b, x, t)
            for layer in range(N_hidden):
                num_rows = w_and_b[layer].shape[0]
                selected_indices = np.random.choice(num_rows, int(np.floor(num_rows*SGD_ratio)))
                w_and_b[layer][selected_indices] -= lmb * cost_grad[layer][selected_indices]
            cost_val = cost(w_and_b, x, t)
            cost_arr[iteration] = cost_val
            print("cost in iteration ", iteration + 1, " is ", cost_val)
    else:
        for iteration in range(n_iter):
            cost_grad = cost_grad_func(w_and_b, x, t)
            for layer in range(N_hidden):
                w_and_b[layer] -= lmb * cost_grad[layer]
            cost_val = cost(w_and_b, x, t)
            cost_arr[iteration] = cost_val
            print("cost in iteration ", iteration + 1, " is ", cost_val)
            
    return w_and_b, cost_arr

if __name__ == '__main__':
    npr.seed(1)
    
    #Hyperparameters
    Nx = 10
    Nt = 10
    n_iter = 20
    lmb = 0.1
    num_neurons = [20, 20]
    
    x = np.linspace(0, 1, Nx)
    t = np.linspace(0, 1, Nt)
    
    t1 = time.time()
    w_and_b, cost_arr = solve_pde_nn(x, t, num_neurons, n_iter, lmb, SGD = True)
    t2 = time.time()
    print(t2-t1)
    
    #outputs
    g_nn = np.zeros((Nx, Nt))
    g_an = np.zeros((Nx, Nt)) #analytic solution
    
    for ix, x_ in enumerate(x):
        for it, t_ in enumerate(t):
            g_nn[ix, it] = g_trial(np.array([x_,t_]), w_and_b)
            g_an[ix, it] = g_analytic(np.array([x_,t_]))

    #saving the hyper parameters and data
    postfix = "_Nx_" + str(Nx) + "_Nt_" + str(Nt) + "_Ni_" + str(n_iter) + "_lmb_" + str(lmb) + "_neurons_"
    for neuron in num_neurons:
        postfix += str(neuron) + "_"
    
    np.save('Nx' + postfix + '.npy', Nx)
    np.save('Nt' + postfix + '.npy', Nt)
    np.save('n_iter' + postfix + '.npy', n_iter)
    np.save('lmb' + postfix + '.npy', lmb)
    np.save('num_neurons' + postfix + '.npy', np.array(num_neurons))
    np.save('g_nn' + postfix + '.npy', g_nn)
    np.save('g_an' + postfix + '.npy', g_an)
    np.save('cost_arr' + postfix + '.npy', cost_arr)
    for layer in range(len(num_neurons)):
        np.save('w_and_b' + postfix + '_layer_' + str(layer+1) + '.npy', w_and_b[layer])
    #w_and_b = np.load('w_and_b.npy', allow_pickle=True)
    np.save('time'+ postfix + '.npy', t2-t1)

    

