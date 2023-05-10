from math import sqrt

# calculate RMSE
def calcRMSE(target, prediction):
    error = []
    for i in range(target.shape[0]):
        error.append(target[i] - prediction[i])
    
    squaredError = []
    for val in error:
        squaredError.append(val * val)

    RMSE_val = sqrt(sum(squaredError) / len(squaredError))
    return RMSE_val 
