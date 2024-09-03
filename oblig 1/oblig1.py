import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Problem 1a)
SF = pd.read_csv("SpotifyFeatures.csv")                                                   # Reads .csv file
print(f"There is {len(SF.axes[0])} rows and {len(SF.axes[1])} columns in .csv file.")     # Prints information about how many rows and columns the .csv file includes


# Problem 1b)
SFr = pd.read_csv("SpotifyFeatures.csv",usecols = ["genre","danceability","energy"]) # Reads .csv file and filters out unnused columns
SFr = SFr.loc[(SFr['genre'] == 'Ska') | (SFr['genre'] == 'Opera')]                   # Filters out unwanted song genre
SFr = SFr.replace({'Opera':1, 'Ska':0})                                              # Renames Opera to value 1, and Ska to value 0 for classification purposes



# Problem 1c)
SFrN = SFr.to_numpy()
test = SFr.drop(columns = ['genre'])                                     # Removes "genre" from dataframe
test = test.to_numpy()                                                   # Converts dataframe to numpy array (matrix)
test2 = SFr.drop(columns = ['danceability','energy'])                    # Removes "danceabiliy" and "energy" from dataframe
test2 = test2.to_numpy().flatten().T                                     # Converts dataframe to numpy array (vector)  
count_test = np.count_nonzero(SFrN[:,0] > 0)
count_test2 = np.count_nonzero(SFrN[:,0] == 0 )


# Test data creation
test_sample = SFr.sample(frac = 0.2, replace = False,random_state = 1)# Creates test sample from dataset
print(test_sample)
test_s_m = test_sample.drop(columns = ['genre'])                         # Removes genre column from dataset to create test training matrix
test_matrix = test_s_m.to_numpy()                                        # Converts test training matrix to np.array
test_s_v = test_sample.drop(columns = ['danceability','energy'])         # Removes dance and energy column from dataset to create label vector
test_vector = test_s_v.to_numpy().ravel().reshape(1,-1)                  # Converts test label vector to np.array

# Training data creation
training_sample = SFr.drop(test_sample.index)                            # Removes test sample from training dataset
training_s_m = training_sample.drop(columns = ['genre'])                 # Removes genre column from dataset to create test training matrix
training_matrix = training_s_m.to_numpy()                                # Converts test training matrix to np.array
training_s_v = training_sample.drop(columns = ['danceability','energy']) # Removes dance and energy column from dataset to create label vector
training_vector = training_s_v.to_numpy().ravel().reshape(1,-1)          # Converts training label vector to np.array

# Used for counting the allocation of labell between training and test data
test_sample_count_opera = np.count_nonzero(test_vector[:,] > 0)
test_sample_count_ska = np.count_nonzero(test_vector[:,] == 0)
training_sample_count_opera = np.count_nonzero(training_vector[:,] > 0 )
training_sample_count_ska = np.count_nonzero(training_vector[:,] == 0)  


# print(len(sample_training), len(sample_test), len(sample_training)+len(sample_test))
# print (training_sample_count_opera, training_sample_count_ska, test_sample_count_opera , test_sample_count_ska)
# print(training_sample_count_opera +training_sample_count_ska, test_sample_count_opera + test_sample_count_ska)


# Problem 2a)
np.random.seed(42)
W = 0.1 * np.random.randn(2)
B = 1
W = W.reshape(-1,1)
training_matrix

def y_hat(training_matrix,W,B):
    return training_matrix * W + B

def loss(y_hat, training_vector):
    m = len(y_hat)
    return 1/m * np.sum(y_hat.T - training_vector)**2

def forward(training_matrix, W, B):
    z = np.dot(W.T, training_matrix.T) + B
    A = sigm(z)
    return A
    
def backward(training_matrix, training_vector, y_hat):
    m = len(y_hat)
    d_W = 1/m * np.sum( -2 * training_matrix.T * (training_vector - y_hat))
    d_B = 1/m * np.sum( -2 * (training_vector.T - y_hat))
    return (d_W, d_B)

    
def update(W, B, d_W, d_b, learning_rate = 0.01):
    W = W - learning_rate * d_W
    B = B - learning_rate * d_b
    return (W, B)
    
def roundValue(A):
    return np.uint8( A > 0.5)

def accuracy(y_hat, training_vector):
    return round(np.sum(y_hat==training_vector) / len(y_hat) * 1000) / 10
    
def sigm(z):
    return 1/(1+np.exp(-z))

def predict(training_matrix,x):
    return np.dot(training_matrix,W)

def cost(training_matrix,training_vector,W):
    prediction = predict(training_matrix,W)
    N = len(training_vector)
    sq_error = (predict(training_matrix,W) - training_vector)**2
    return (1/(2*N))*sq_error.sum()

iter = 1000
lr = 0.01
losses, acces = [], []
for i in range(iter):
    A = forward(training_matrix, W, B)
    l = loss(training_vector, A)
    y_hat = roundValue(A)
    acc = accuracy(y_hat, training_vector.T)
    d_W, d_B = backward(training_matrix, training_vector, A)
    W, B = update(W, B, d_W, d_B, learning_rate=lr)
    losses.append(l)
    acces.append(acc)
    if i % 1000 == 0:
        print('loss:', l, f'\taccuracy: {accuracy(y_hat, training_matrix.T)}%') 



# i = 1
# iter = 10
# learning_rate = 0.1
# while i <= iter:
#     z = np.dot(training_matrix, W)
#     pred = sigm(z)
#     error = training_vector - pred
#     grad = (training_matrix.T * (pred - training_vector)) / len(training_vector) * learning_rate
#     # print(grad) 
#     # x = np.subtract(x,grad)
#     # x -= learning_rate * grad
#     i += 1
#     print(i)
# print(z,pred,grad)

print("whhhhat?")