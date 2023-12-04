import numpy as np
from numba import njit

# Block 1: Definition of Numba-optimized functions
# These functions are optimized for performance using Numba's JIT compilation.

# Function 'lm': Performs logistic map calculations using Numba for speed optimization.
@njit('float64[:,:](float64[:],float64[:],int64,int64)', parallel=True)
def mod_logistic_map(x0, r, L, T):
    ''' Iterates the logistic map T times from n initial conditions x0 and n r-values r.
        Returns the last L results. '''
    a = x0.size    # Number of initial conditions
    ret = np.zeros((a, L))      
    ret[:,0] = x0
    for i in range(ret.shape[0]):
        for t in range(1, T):
            t = np.mod(t, ret.shape[1])
            ret[i,t] = r[i]*ret[i,t-1]*(1-ret[i,t-1])
    return ret


@njit('float64[:,:](float64[:],float64[:],int64,int64)', parallel=True)
def mod_rsinpix_map(x0, r, L, T):
    ''' Iterates the r*sin(pi*x) map T times from n initial conditions x0 and n r-values r.
        Returns the last L results. '''
    a = x0.size    # Number of initial conditions
    ret = np.zeros((a, L))      
    ret[:,0] = x0
    for i in range(ret.shape[0]):
        for t in range(1, T):
            t = np.mod(t, ret.shape[1])
            ret[i,t] = r[i]*np.sin(np.pi*ret[i,t-1])
    return ret



# Function 'uni': Calculates the uniqueness of elements in each row of an array.
@njit('float64[:](float64[:,:])')
def uni(arr):
    ''' Returns the log number of unique elements in each row of arr. '''
    ret = np.zeros(arr.shape[0])
    for i, el in enumerate(arr):
        ret[i] = np.log2(np.unique(el).size)
    return ret

# Block 2: Main function 'main'
# This is the core of the script, where the bifurcation analysis is performed.
def main():
    map_function = mod_logistic_map
    N = 7 # Number of bifurcation points
    T = 2**25 # Total iterations
    L = 2**5 # Iterations to save
    B = 10  # Number of r values to search in every iteration
    LB = 3.44   # Absolute lower bound of r
    UB = 3.57    # Absolute upper bound of r
    res = np.zeros(N)
    ini = np.random.rand(B)     # First initial conditions
    R = np.linspace(LB, UB, B)  # First searching values of R
    r = np.linspace(LB, UB, B)
    ini = map_function(ini, R, L, T)[:,-1]    # Run trajectories for some time to skip transients
    roun = 8        # Round to this decimal
    lim = 5         # Number of refinement steps

    # Main loop for bifurcation analysis
    for i in range(1, N+1):
        L = 2**(i+3) # Iterations to save
        ct = 0
        it = 0
        if i > 7: lim += 2
        while ct < lim:
            # Logistic map calculations and rounding
            ret = map_function(ini, r, L, T)      # Run trajectories in the current attractor
            ini = ret[:,-1]
            ret = np.round(ret, roun)   # Round trajectories to precision 
            re = uni(ret)               # Check how many unique points each trajectory has
            # Handling indices and adjusting ranges
            try:
                idx1 = np.where(re == i)[0][-1]     #  find the last trajectory which has i unique points (trajectories are ordered by r!)
            except IndexError:
                # Error handling and parameter adjustments
                it += 1
                print('one not found')
                print(f'Searching in r={r}')
                print(f'Number of unique points per trajectory: {re}')                
                print(f'Number of unsuccessful consecutive iterations: {it}')
                print('Increasing precision...')
                if roun > 15:
                    # Further adjustments if necessary
                    idx1 = np.where(re <= i)[0][-1]
                    idx2 = np.where(re > i)[0][0]
                    lb = r[idx1]
                    ub = r[idx2]
                    r = np.linspace(lb, ub, B)
                else:
                    roun = roun + 1
                continue
            # Determining lower and upper bounds
            idx2 = np.where(re >= i+1)[0][0]        # find the first trajectory which has at least i+1 unique points
            if re[idx2] > i+1:  	    # if there the difference in unique points is more than 1, search in that area
                lb = r[idx1]
                ub = r[idx2]
                r = np.linspace(lb, ub, B)
                it += 1 
                print(f'Searching in r={r}')
                print(f'Number of unique points per trajectory: {re}')
                print(f'Number of unsuccessful consecutive iterations: {it}')
                if it > 3:
                    roun = roun + 1
                continue
            
            lb = r[idx1]            # Refine the search space
            ub = r[idx2]
            r = np.linspace(lb, ub, B)
            print(f'Number of total iterations for bifurcation {i}: {ct+1}')
            ct = ct + 1
        idx1 = np.where(re == i)[0][-1]
        res[i-1] = r[idx1]
        idx2 = np.where(re == i+1)[0][0]
        LB = r[-1]
        # UB = 3.57
        r = np.linspace(LB, UB, B)
        print(f'Found bifurcation {i} \n')
        # Saving results to a file
        np.save('bifurcations.npy', res)
    return res

# Block 3: Execution of the main function and additional calculations
# This block is executed if the script is run as a
# main module, not when imported as a module.

if __name__ == '__main__':
    # Calling the main function and storing its result
    res = main()
    res = np.load('lm_bifurcations.npy')

    c = (res[1:-1] - res[:-2])/(res[2:] - res[1:-1])
    print(f'Bifurcation points: {res}')
    print(f'Bifurcation point distance ratio: {c}')
    # This line prints out the calculated ratios, which can be used to analyze the dynamics of the system.

# Optional Debugging Line
# Uncomment the line below to print the logarithm of the unique size of a 2D array along axis 2.
# Useful for debugging or additional analysis.
# print(np.log2(np.unique(ret,axis=2).size))

