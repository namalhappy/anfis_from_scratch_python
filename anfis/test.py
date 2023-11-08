import numpy as np
import myANFIS as anfis
from sklearn.preprocessing import MinMaxScaler


data = np.genfromtxt('input/iris.csv', delimiter=',')  
# Divide data into input and output
inputs = data[:, :-1]  # All columns except the last one are inputs
output = data[:, -1:]  # The last column is the output
ndata = data.shape[0]  # Data length

    
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_input = scaler.fit_transform(inputs)


# Settings for ANFIS model
epoch_n = 10
mf = 3
step_size = 0.1
decrease_rate = 0.9
increase_rate = 1.1

# ANFIS train 
bestnet, y_myanfis, RMSE = anfis.myanfis(data, scaled_input, epoch_n, mf, step_size, decrease_rate, increase_rate)
y_myanfis = anfis.evalmyanfis(bestnet, scaled_input)

anfis_predictions = y_myanfis

# For classification problem ( Round outputs to int)
anfis_predictions = np.round(anfis_predictions).astype(int)

# Calculate the RMSE
rmse = anfis.calc_rmse(output,anfis_predictions)

msg = f'Total RMSE error myanfis: {rmse:.2f}'
print(msg)  # Print the message

anfis.plot_Nodes(bestnet)

anfis.plot_mf(bestnet, data)

anfis.plot_predictions(output,anfis_predictions)

anfis.plot_r2(output,anfis_predictions)

print("Welcome !")