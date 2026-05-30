import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_split_data(num_clients=3):
    """
    Loads the MNIST Digits dataset, normalizes it, and splits it evenly
    among a specified number of simulated IoT clients.
    """
    print("Loading digits dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Preprocess: scale data for better convergence in Logistic Regression
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the dataset into a train and test set (for global evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Split the training data among multiple clients
    # We will just divide arrays into `num_clients` portions
    client_X = np.array_split(X_train, num_clients)
    client_y = np.array_split(y_train, num_clients)
    
    clients_data = []
    for i in range(num_clients):
        clients_data.append((client_X[i], client_y[i]))
        print(f"Client {i+1} received {len(client_X[i])} training samples.")
        
    global_test_data = (X_test, y_test)
    classes = np.unique(y) # Digits 0-9
    
    return clients_data, global_test_data, classes
