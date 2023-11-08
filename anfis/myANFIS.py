import numpy as np
import csv
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.metrics import mean_squared_error, r2_score

# Custom Gaussian bell-shaped membership function
def gbellmf(x, params):
    a, b, c = params
    return 1 / (1 + ((x - c) / a) ** (2 * b))
    
    
def calculate_output1(mynet):
    mparams = mynet['mparams']

    for i in range(mynet['ni']):
        for j in range(mynet['mf']):
            ind = mynet['ni'] + i * mynet['mf'] + j

            x = mynet['nodes'][i]
            a = mparams[i * mynet['mf'] + j, 0]
            b = mparams[i * mynet['mf'] + j, 1]
            c = mparams[i * mynet['mf'] + j, 2]

            tmp1 = (x - c) / a
            if tmp1 == 0:
                tmp2 = 0
            else:
                tmp2 = (tmp1 ** 2) ** b
            mynet['nodes'][ind] = 1 / (1 + tmp2)
    mynet['nodes'] = mynet['nodes']
    return mynet
    
def calculate_output2(mynet):
    st = mynet['ni'] + mynet['ni'] * mynet['mf']
    for i in range(st, st + mynet['nc']):
        I = [idx for idx, val in enumerate(mynet['config'][:, i]) if val == 1]
        tmp = 1
        for idx in I:
            tmp *= mynet['nodes'][idx]
        mynet['nodes'][i] = tmp
    return mynet

def calculate_output3(mynet):
    st = mynet['ni'] + mynet['ni'] * mynet['mf'] + mynet['nc']    
    for i in range(st , st + mynet['nc'] ):
        I = [idx for idx, val in enumerate(mynet['config'][:, i]) if val == 1]
        denom = sum([mynet['nodes'][idx] for idx in I])
        mynet['nodes'][i] = mynet['nodes'][i - mynet['nc']] / denom
    return mynet

def calculate_output4(mynet):
    st = mynet['ni'] + mynet['ni'] * mynet['mf'] + 2 * mynet['nc']
    inp = mynet['nodes'][:mynet['ni']]
    kparam = mynet['kparams']
    
    for i in range(mynet['nc']):
        wn = mynet['nodes'][i + st - mynet['nc']]
        mynet['nodes'][i + st] = wn * (np.sum(kparam[i, :-1] * inp) + kparam[i, -1])
    
    # print(mynet['nodes'])
    # print(np.shape(mynet['nodes']))
    # exit()
    return mynet

def calculate_output5(mynet):
    mynet['nodes'][-1] = sum(mynet['nodes'][-mynet['nc']-1:-1])
    # print(np.shape(mynet['nodes']))
    # print(mynet['nodes'])
    # exit()
    return mynet


def get_kalman_data(mynet, target):
    
    kalman_data = np.zeros(((mynet['ni'] + 1) * mynet['nc'] + 1, 1))
    
    st = mynet['ni'] + mynet['ni'] * mynet['mf'] + mynet['nc']
    j = 0
    # print(st + mynet['nc'] + 1)
    for i in range(st, st + mynet['nc']):
        for k in range(mynet['ni']):
            kalman_data[j] = mynet['nodes'][i] * mynet['nodes'][k]
            j += 1
        kalman_data[j] = mynet['nodes'][i]
        j += 1

    kalman_data[j] = target
    return kalman_data

def mykalman(mynet, kalman_data, k):
    k_p_n = (mynet['ni'] + 1) * mynet['nc']

    alpha = 1000000
    # print(k)
    if k == 0:
        mynet['P'] = np.zeros((k_p_n, 1))
        mynet['S'] = alpha * np.eye(k_p_n)
    
    # print(mynet['P'])
    # print(mynet['S'])
        
    # exit()
    x = kalman_data[:-1]
    y = kalman_data[-1]
           
    
    x_temp = x.flatten()
    tmp1 = np.dot(x_temp, mynet['S'])
    tmp1 = tmp1[:, np.newaxis]

    denom = 1 + np.sum(tmp1 * x)
    tmp1 = np.dot(mynet['S'], x_temp)
    tmp1 = tmp1[:, np.newaxis]
        

    tmp2 = np.dot(x_temp, mynet['S'])
    tmp2 = tmp2[:, np.newaxis]
    tmp_m = np.outer(tmp1, tmp2)
    tmp_m = -1 / denom * tmp_m
    mynet['S'] = mynet['S'] + tmp_m
    
    diff = y - np.sum(x * mynet['P'])
    tmp1 = diff * np.dot(mynet['S'], x_temp)
    tmp1 = tmp1[:, np.newaxis]
    
    mynet['P'] = mynet['P'] + tmp1
    mynet['kparams'] = mynet['P'].reshape(( mynet['nc'],mynet['ni'] + 1))
    # mynet['kparams'] = mynet['P'].reshape(8,4)

    return mynet

def clear_de_dp(mynet):
    mynet['mparam_de_do'] = np.zeros((mynet['ni'] * mynet['mf'], 3))
    mynet['kparam_de_do'] = np.zeros((mynet['nc'], mynet['ni'] + 1))

    return mynet

def calculate_de_do(mynet, de_dout):
    mynet['de_do'] = np.zeros_like(mynet['nodes'])
    mynet['de_do'][-1] = de_dout
    tmp2 = []

    for i in range(len(mynet['nodes']) - 2, mynet['ni']-1, -1):
        # print("i = ", i)
        # print(len(mynet['nodes']) - 2)
        # print(mynet['ni']+1)
        
        de_do = 0
        II = np.where(mynet['config'][i, :] == 1)[0]
        # print(mynet['config'][i, :])
        # print("II =", II)
        I = np.where(II > i)[0]
        # print("I =",I)      
        for j in I:
            jj = II[j]
            tmp1 = mynet['de_do'][jj]            
            tmp2 = derivative_o_o(mynet, i, jj) 
            # print("tmp2 =",tmp2)
            # ui = input()            
            de_do = de_do + tmp1 * tmp2
        mynet['de_do'][i] = de_do
        # print(tmp2)
        # ui = input()
        # print(I, mynet['de_do'][i], tmp1, tmp2)
        # exit()
    # print(np.shape(mynet['de_do']))
    # print((mynet['de_do']))
    # exit()
    return mynet

def derivative_o_o(mynet, i, j):
    if i > mynet['ni'] + mynet['ni'] * mynet['mf'] +  2 * mynet['nc']-1:
        # print(i, "\t In 1", mynet['ni'] + mynet['ni'] * mynet['mf'] + 2 * mynet['nc']-1)
        return 1
    elif i > mynet['ni'] + mynet['ni'] * mynet['mf'] + mynet['nc']-1:
        # print(i, "\t In 2", mynet['ni'] + mynet['ni'] * mynet['mf'] + mynet['nc'])
        return do4_do3(mynet, i, j)
    elif i > mynet['ni'] + mynet['ni'] * mynet['mf']-1:
        # print(i, "\t In 3", mynet['ni'] + mynet['ni'] * mynet['mf'])
        return do3_do2(mynet, i, j)
    elif i > mynet['ni']-1:
        # print(i, "\t In 4",mynet['ni'])
        # print(mynet['nodes'][j])
        # print(mynet['nodes'][i])
        return mynet['nodes'][j] / mynet['nodes'][i]

def do4_do3(mynet, i, j):
    kparam = mynet['kparams']
    inp = mynet['nodes'][:mynet['ni']]
    # print("inp =", inp)    
    jj = j - mynet['ni'] - mynet['ni'] * mynet['mf'] - 2 * mynet['nc']   
    # print("jj =", jj)  
    #print("inp and jj - ", inp,jj )
    return np.sum(kparam[jj, :-1] * inp) + kparam[jj, -1]

def do3_do2(mynet, i, j):
    II = np.where(mynet['config'][:, j] == 1)[0]
    I = np.where(II < j)[0]
    total = np.sum(mynet['nodes'][II[I]])
    # print("total = ",total)
    # print("j = ",j)
    # print("mynet['nc'] = ",mynet['nc'])
    if j - i == mynet['nc']:
        # print(("r1 = ",total - mynet['nodes'][i]) / (total**2))
        return (total - mynet['nodes'][i]) / (total**2)
        
    else:
        # print("r2 = ",-mynet['nodes'][j - mynet['nc']] / (total**2))
        return -mynet['nodes'][j - mynet['nc']] / (total**2)
        


def update_de_do(mynet):
    s = 0
    for i in range(mynet['ni'], mynet['ni'] + mynet['ni'] * mynet['mf']):
        for j in range(3):
            do_dp = dmf_dp(mynet, i, j)
            mynet['mparam_de_do'][s, j] = mynet['mparam_de_do'][s, j] + mynet['de_do'][i] * do_dp
        s += 1

    s = 0
    for i in range(1 + mynet['ni'] + mynet['ni'] * mynet['mf'] + 2 * mynet['nc'], len(mynet['config'])):
        for j in range(mynet['ni'] + 1):
            do_dp = dconsequent_dp(mynet, i, j)
            mynet['kparam_de_do'][s, j] = mynet['kparam_de_do'][s, j] + mynet['de_do'][i] * do_dp
        s += 1

    return mynet

def dmf_dp(mynet, i, j):
    I = np.where(mynet['config'][:, i] == 1)[0]
    x = mynet['nodes'][I]
    a = mynet['mparams'][i - mynet['ni'], 0]
    b = mynet['mparams'][i - mynet['ni'], 1]
    c = mynet['mparams'][i - mynet['ni'], 2]
    tmp1 = (x - c) / a
    if tmp1 == 0:
        tmp2 = 0
    else:
        tmp2 = (tmp1**2)**b
    denom = (1 + tmp2) * (1 + tmp2)
    if j == 0:
        tmp = (2 * b * tmp2) / (a * denom)
    elif j == 1 and tmp1 == 0:
        tmp = 0
    elif j == 1 and tmp1 != 0:
        tmp = (-np.log(tmp1**2) * tmp2) / denom
    elif j == 2 and x == c:
        tmp = 0
    elif j == 2 and x != c:
        tmp = (2 * b * tmp2) / ((x - c) * denom)
    return tmp

def dconsequent_dp(mynet, i, j):
    wn = mynet['nodes'][i - mynet['nc']-1]
    inp = mynet['nodes'][:mynet['ni']]
    inp = np.append(inp, 1)
    return wn * inp[j]

def update_step_size(mynet, error_array, iter, step_size, decrease_rate, increase_rate):
    if check_decrease_ss(error_array, mynet['last_decrease_ss'], iter):
        step_size = step_size * decrease_rate
        mynet['last_decrease_ss'] = iter
    elif check_increase_ss(error_array, mynet['last_increase_ss'], iter):
        step_size = step_size * increase_rate
        mynet['last_increase_ss'] = iter

    return mynet, step_size

def check_decrease_ss(error_array, last_change, current):
    if current - last_change < 4:
        return False
    elif (error_array[current] < error_array[current - 1] and
          error_array[current - 1] > error_array[current - 2] and
          error_array[current - 2] < error_array[current - 3] and
          error_array[current - 3] > error_array[current - 4]):
        return True
    else:
        return False

def check_increase_ss(error_array, last_change, current):
    if current - last_change < 4:
        return False
    elif (error_array[current] < error_array[current - 1] and
          error_array[current - 1] < error_array[current - 2] and
          error_array[current - 2] < error_array[current - 3] and
          error_array[current - 3] < error_array[current - 4]):
        return True
    else:
        return False

def update_parameter(mynet, step_size):
    tmp = mynet['mparam_de_do']
    tmp = tmp * tmp
    norm_factor = np.sqrt(np.sum(tmp))
    mynet['mparams'] = mynet['mparams'] - step_size * mynet['mparam_de_do'] / norm_factor
    return mynet

def evalmyanfis(mynet, inputs):
    anfis_output = np.zeros((inputs.shape[0], 1))
    for i in range(inputs.shape[0]):
        mynet['nodes'][:mynet['ni']] = inputs[i]
        # mynet['nodes'][:mynet['ni']] = inputs[j].T.tolist()
        mynet = calculate_output1(mynet)
        mynet = calculate_output2(mynet)
        mynet = calculate_output3(mynet)
        mynet = calculate_output4(mynet)
        mynet = calculate_output5(mynet)

        anfis_output[i, 0] = mynet['nodes'][-1]
    
    return anfis_output
        

def myanfis(data, inputs, epoch_n, mf, step_size, decrease_rate, increase_rate):
    # Divide data into input and output    
    output = data[:, -1:]  # The last column is the output

    ndata = data.shape[0]  # Data length

    # Define minimum and maximum of input to determine initial membership functions
    mn = np.min(inputs, axis=0)
    mx = np.max(inputs, axis=0)
    mm = mx - mn
    ni = inputs.shape[1]  # Number of inputs
    nc = mf**ni  # Number of rules
    Node_n = ni + ni * mf + 3 * nc + 1  # Total number of nodes
    
    min_RMSE = 999999999999  # Define minimum RMSE

    # Define initial membership functions
    mparams = []
    for i in range(ni):
        tmp = np.linspace(mn[i], mx[i], mf)
        mparams.extend(np.column_stack([np.full(mf, mm[i] / 6), np.full(mf, 2), tmp]))

    # Define initial Kalman parameters with all zeros
    kparams = np.zeros((nc, ni + 1))
    
    # Create connection matrix and node array
    config = np.zeros((Node_n, Node_n))
    nodes = np.zeros(Node_n)

    # Inputs - layer1 connections
    st = ni
    for i in range(ni):
        config[i, st : st + mf] = 1
        st = st + mf

    # Layer1 - Layer2 connections
    st = ni + ni * mf 
    if ni == 2:
        for i in range(mf):
            for j in range(mf):
                config[ni + i, st] = 1
                config[ni + mf + j, st] = 1
                st = st + 1
    elif ni == 3:
        for i in range(mf):
            for j in range(mf):
                for k in range(mf):
                    config[ni + i, st] = 1
                    config[ni + mf + j, st] = 1
                    config[ni + 2 * mf + k, st] = 1
                    st = st + 1
    elif ni == 4:
        for i in range(mf):
            for j in range(mf):
                for k in range(mf):
                    for l in range(mf):
                        config[ni + i, st] = 1
                        config[ni + mf + j, st] = 1
                        config[ni + 2 * mf + k, st] = 1
                        config[ni + 3 * mf + l, st] = 1
                        st = st + 1
    elif ni == 5:
        for i in range(mf):
            for j in range(mf):
                for k in range(mf):
                    for l in range(mf):
                        for m in range(mf):
                            config[ni + i, st] = 1
                            config[ni + mf + j, st] = 1
                            config[ni + 2 * mf + k, st] = 1
                            config[ni + 3 * mf + l, st] = 1
                            config[ni + 4 * mf + m, st] = 1
                            st = st + 1

    # Layer2 - Layer3 connections
    for i in range(nc):
        for j in range(nc):
            config[ni + ni * mf + i, ni + ni * mf + nc + j] = 1

    # Layer3 - Layer4 connections
    for i in range(nc):
        config[ni + ni * mf + nc + i, ni + ni * mf + 2 * nc + i] = 1

    # Layer4 - Layer5 connections
    for i in range(nc):
        config[ni + ni * mf + 2 * nc + i, -1] = 1

    # Inputs - Layer4 connections
    for i in range(ni):
        for j in range(nc):
            config[i, ni + ni * mf + 2 * nc + j] = 1

    mynet = {
        "config": config,
        "mparams": np.array(mparams),
        "kparams": kparams,
        "nodes": nodes,
        "ni": ni,
        "mf": mf,
        "nc": nc,
        "last_decrease_ss": 1,
        "last_increase_ss": 1,
    }
    
    num_nodes = len(mynet['nodes'])  # You need to determine this
    
    layer_1_to_3_output = np.zeros((num_nodes, ndata))  # Initialize with the correct shape
    

    for iter in range(0, epoch_n):
        for j in range(ndata):

            mynet['nodes'] = mynet['nodes'].flatten()
            mynet['nodes'][:mynet['ni']] = inputs[j]

            mynet = calculate_output1(mynet)
            mynet = calculate_output2(mynet)
            mynet = calculate_output3(mynet)
                     
            # Save outputs of layer 1 to 3
            layer_1_to_3_output[:, j] = mynet['nodes'].flatten() 
            # print((mynet['nodes'].flatten()))
            # print((layer_1_to_3_output))
            # exit()   
            kalman_data = get_kalman_data(mynet, output[j])
            
            # exit()
            mynet = mykalman(mynet, kalman_data, j)
            
        mynet = clear_de_dp(mynet)
    
        # print(np.shape(layer_1_to_3_output))
        # print((layer_1_to_3_output))
        # file_path = "layer_1_to_3_output.csv"

        ## Use numpy.savetxt to save the array to a CSV file
        # np.savetxt(file_path, layer_1_to_3_output, delimiter=',')
        # exit()
        anfis_output = np.zeros((ndata, 1))
        RMSE = np.zeros((epoch_n, 1))  # Initialize RMSE as a NumPy array of zeros

        for j in range(ndata):
            mynet['nodes'] = layer_1_to_3_output[:, j]   
            
            
            mynet = calculate_output4(mynet)
            mynet = calculate_output5(mynet)
        
            anfis_output[j, 0] = mynet['nodes'][-1]
            target = output[j]
            
            de_dout = -2 * (target - anfis_output[j, 0])
        
            mynet = calculate_de_do(mynet, de_dout)
            
            mynet = update_de_do(mynet)
            

        diff = anfis_output - output
        total_squared_error = np.sum(diff**2)
        RMSE[iter - 1, 0] = np.sqrt(total_squared_error / ndata)
        print(f'{iter}. RMSE error: {RMSE[iter - 1, 0]}')
        # input()
        if RMSE[iter - 1, 0] < min_RMSE:
            bestnet = mynet
            min_RMSE = RMSE[iter - 1, 0]

        mynet = update_parameter(mynet, step_size)
        mynet, step_size = update_step_size(mynet, RMSE, iter, step_size, decrease_rate, increase_rate)

# At this point, bestnet contains the trained ANFIS model, and RMSE contains the RMSE values for each epoch.
    for j in range(ndata):       
        # print(j)
        # mynet['nodes'][:mynet['ni']].flatten()
        # print(np.shape(mynet['nodes'][:mynet['ni']]))
        # print(np.shape(inputs[j]))
        
        mynet['nodes'][:mynet['ni']] = inputs[j]
        # mynet['nodes'][:mynet['ni']] = inputs[j].T.tolist()
        mynet = calculate_output1(mynet)
        mynet = calculate_output2(mynet)
        mynet = calculate_output3(mynet)
        mynet = calculate_output4(mynet)
        mynet = calculate_output5(mynet)

        anfis_output[j, 0] = mynet['nodes'][-1]
    
    return bestnet,anfis_output,RMSE

def plot_Nodes(mynet):
    # Plot the Node Connections
    plt.figure()
    plt.imshow(mynet['config'], aspect='auto', cmap='cool')
    plt.title('Node Connections')
    plt.show()

def plot_mf(mynet, data):
    plt.figure(figsize=(12, 4))  # Create a single figure for all plots
    # print(mynet['mparams'])
    k = 0
    for i in range(0,mynet['ni']*mynet['mf'],mynet['mf']):
        # print(k)
        plt.subplot(2, 2, k+1)  # Create subplots for each input variable
        plt.title(f'Input {k+1}')
        plt.xlabel('X')
        plt.ylabel('Degree of Membership')
        min_val = np.min(data[:, k])
        max_val = np.max(data[:, k])
        step = 0.1
        x = np.arange(min_val, max_val, step)
        x = np.arange(-10, 10, step)
        
        for j in range(mynet['mf']):  
            mf_x = fuzz.gbellmf(x, mynet['mparams'][i+j][0], mynet['mparams'][i+j][1], mynet['mparams'][i+j][2])
            plt.plot(x, mf_x, label=f'Membership Function {j+1}')
        k = k+1
        plt.tight_layout()
        plt.legend()
    plt.show()
    
def plot_predictions(actual_output,anfis_predictions):
    plt.figure()
    plt.plot(actual_output, 'b*', label='Actual Output')
    plt.plot(anfis_predictions, 'r-', linewidth=0.5, label='ANFIS Prediction')
    plt.xlabel('Data Point')
    plt.ylabel('Output Value')
    plt.legend()
    plt.show()
    

def calc_rmse(x,y):
    # Calculate Root Mean Squared Error (RMSE)
    mse = mean_squared_error(x, y)
    rmse = np.sqrt(mse)
    
    return rmse

    
def calc_r2(x,y):
    # Calculate R-squared
    r_squared = r2_score(x, y)
    
    return r_squared
    
def plot_r2(x,y):
    
    r_squared = r2_score(x, y)
     # Create a scatter plot for the data points
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label=f'R-squared = {r_squared:.4f}')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Add the identity line
    plt.xlabel('Actual Output')
    plt.ylabel('Predicted Output')
    plt.title("Correlation_Coefficient (R2)")
    plt.legend(loc='upper left')
    plt.tight_layout()
    # Show the plot
    plt.show
    
    
    