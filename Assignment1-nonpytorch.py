import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Data Preparation
num_header_lines = 24
file_path = 'C:\\Users\\stellan.lange\\JVarback-Learning-systems-ANN\\Diabetic.txt'
df = pd.read_csv(file_path, skiprows=num_header_lines, sep=',')

train_df_portion = int(len(df) * 0.75)
test_df_portion = int(len(df) * 0.15)
eval_df_portion = int(len(df) * 0.10)

train_df = df.sample(n=train_df_portion, random_state=42)
non_training_selected_df = df.drop(train_df.index)
test_df = non_training_selected_df.sample(n=test_df_portion, random_state=42)
eval_df = non_training_selected_df.drop(test_df.index).sample(n=eval_df_portion, random_state=42)

# Data Loaders
X_train = torch.tensor(train_df.iloc[:, :-1].values, dtype=torch.float32)
y_train = torch.tensor(train_df.iloc[:, -1].values, dtype=torch.float32)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

X_eval = torch.tensor(eval_df.iloc[:, :-1].values, dtype=torch.float32)
y_eval = torch.tensor(eval_df.iloc[:, -1].values, dtype=torch.float32)
eval_dataset = TensorDataset(X_eval, y_eval)
val_loader = DataLoader(eval_dataset, batch_size=32, shuffle=True)

X_test = torch.tensor(test_df.iloc[:, :-1].values, dtype=torch.float32)
Y_test = torch.tensor(test_df.iloc[:, -1].values, dtype=torch.float32)
test_dataset = TensorDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Neural Network Setup
def sigmoid_activation(x, derivative=False):
    sigmoid = 1 / (1 + np.exp(-x))
    if derivative:
        return sigmoid * (1 - sigmoid)
    else:
        return sigmoid

def init_params(input_size, hidden_size, output_size):
    hidden_weights = np.random.randn(input_size, hidden_size) * 0.1
    output_weights = np.random.randn(hidden_size, output_size) * 0.1
    hidden_biases = np.zeros((1, hidden_size))
    output_biases = np.zeros((1, output_size))
    return hidden_weights, output_weights, hidden_biases, output_biases

def forward_prop(x_train, hidden_weights, output_weights, hidden_biases, output_biases):
    hidden_layer_input = np.dot(x_train, hidden_weights) + hidden_biases
    hidden_activation = sigmoid_activation(hidden_layer_input)
    output_layer_input = np.dot(hidden_activation, output_weights) + output_biases
    output_activation = sigmoid_activation(output_layer_input)
    return hidden_activation, output_activation

def backpropagation(X_train, y_train, hidden_activation, output_activation, output_weights):
    data_info_all_examples = len(X_train)  # Number of training examples
    d_comb_output = output_activation - y_train.reshape(-1, 1)  # Reshape y_train to match output_activation
    d_weight_output = np.dot(hidden_activation.T, d_comb_output) / data_info_all_examples
    d_bias_output = np.sum(d_comb_output, axis=0, keepdims=True) / data_info_all_examples

    # Propagate the error to the hidden layer
    error_hidden = np.dot(d_comb_output, output_weights.T)

    # Adjust error at hidden layer based on the derivative of the activation function
    activation_at_hidden = sigmoid_activation(hidden_activation, derivative=True)
    error_hidden *= activation_at_hidden  # Adjusted error at hidden layer

    d_weight_hidden = np.dot(X_train.T, error_hidden) / data_info_all_examples
    d_bias_hidden = np.sum(error_hidden, axis=0, keepdims=True) / data_info_all_examples

    return d_weight_hidden, d_bias_hidden, d_weight_output, d_bias_output


# Hyperparameters
input_size = 19
hidden_size = 2048
output_size = 1
learning_rate = 0.5

# Initialize network parameters
hidden_weights, output_weights, hidden_biases, output_biases = init_params(input_size, hidden_size, output_size)

# Lists to store losses
train_losses = []
val_losses = []
test_losses = []
total_test_accuracy = 0

# Training loop
num_epochs = 25
for epoch in range(num_epochs):
    epoch_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.numpy()  # Convert to NumPy array
        y_batch = y_batch.numpy()  # Convert to NumPy array

        # Forward propagation
        hidden_activation, output_activation = forward_prop(X_batch, hidden_weights, output_weights, hidden_biases, output_biases)

        # Convert output_activation to NumPy array for loss computation
        output_activation_np = output_activation

        # Compute loss (mean squared error)
        loss = np.mean((y_batch - output_activation_np) ** 2)
        epoch_loss += loss

        # Backward propagation
        d_weight_hidden, d_bias_hidden, d_weight_output, d_bias_output = backpropagation(X_batch, y_batch, hidden_activation, output_activation_np, output_weights)

        # Update parameters
        hidden_weights -= learning_rate * d_weight_hidden
        hidden_biases -= learning_rate * d_bias_hidden
        output_weights -= learning_rate * d_weight_output
        output_biases -= learning_rate * d_bias_output

    train_losses.append(epoch_loss / len(train_loader))

    # Validation loss
    val_loss = 0
    for x_val, y_val in val_loader:
        x_val, y_val = x_val.numpy(), y_val.numpy()
        _, output_val = forward_prop(x_val, hidden_weights, output_weights, hidden_biases, output_biases)
        val_loss += np.mean((y_val - output_val) ** 2)
    val_losses.append(val_loss / len(val_loader))

    # Test loss and accuracy
    test_loss = 0
    correct = 0
    total = 0
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.numpy(), y_test.numpy()
        _, output_test = forward_prop(x_test, hidden_weights, output_weights, hidden_biases, output_biases)
        test_loss += np.mean((y_test - output_test) ** 2)
        
        predicted = output_test > 0.5  # Thresholding at 0.5 for binary classification
        correct += np.sum(predicted.flatten() == y_test)
        total += y_test.shape[0]

    test_losses.append(test_loss / len(test_loader))
    epoch_accuracy = 100 * correct / total
    total_test_accuracy += epoch_accuracy

    accuracy = 100 * correct / total
    print('Epoch [{}/{}], Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(epoch+1, num_epochs, epoch_loss / len(train_loader), accuracy))

average_test_accuracy = total_test_accuracy / num_epochs
print('Total Average Test Accuracy after all epochs: {:.2f}%'.format(average_test_accuracy))

# Plotting the Losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), train_losses, label='Train')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation')
plt.plot(range(1, num_epochs+1), test_losses, label='Test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training, Validation and Test Loss Over Epochs')
plt.legend()
plt.show()
