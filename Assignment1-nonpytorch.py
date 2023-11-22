import numpy as np
import pandas as pd
import math 
import random as rand

# Data Preparation
num_header_lines = 24
file_path = 'C:\\Users\\stellan.lange\\OneDrive - NIOUSA\\Desktop\\3Dtraffic\\Diabetic.txt'
df = pd.read_csv(file_path, skiprows=num_header_lines, sep=',')

train_df_portion = int(len(df) * 0.75)
test_df_portion = int(len(df) * 0.15)
eval_df_portion = int(len(df) * 0.10)

train_df = df.sample(n=train_df_portion, random_state=42)
non_training_selected_df = df.drop(train_df.index)
test_df = non_training_selected_df.sample(n=test_df_portion, random_state=42)
eval_df = non_training_selected_df.drop(test_df.index).sample(n=eval_df_portion, random_state=42)

class ANN():
    def __init__(self, size_of_input, size_of_output, size_of_hidden_layers, neurons = None, weights = None, biases = None):
        self.weights = []
        self.biases = []
        self.layers = []
        self.size_of_input = size_of_input
        self.size_of_output = size_of_output
        self.size_of_hidden_layers = size_of_hidden_layers
        #self.layers = 

        self.architecture()


    
    
    
    def architecture(self): 
        bias_vector = np.zeros
        weight_matrix = np.zeros

        for i in range(self.layers): 
            rand(self.size_of_input, self.size_of_output)

            for j in range(self.size_of_hidden_layers):
                rand(self.size_of_input, self.size_of_output)

    
    def activation_function_sigmoid(x): 
        s = 1/(1+math.e**(-x))
        ds = s * (1-s) 
    
        return ds 


    def backpropogation():
        pass



    def l1_numpy(y, yhat): #vector of size m (predicted labels) & #vector of size m (true labels)
        loss = sum(y-yhat)
    
        return loss
    
    



