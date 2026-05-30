import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_split_mnist_data(num_clients=5):
    """
    Loads the massive MNIST-784 dataset (70,000 images).
    Splits exactly 60,000 for training and 10,000 for testing.
    Divides the 60,000 training images perfectly among the clients.
    """
    print("Loading 70,000 MNIST dataset from OpenML... (This may take a minute)")
    
    # We use parser='liac-arff' and as_frame=False to prevent Pandas dependency errors
    mnist = fetch_openml('mnist_784', version=1, cache=True, parser='liac-arff', as_frame=False)
    
    # Convert types
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int')
    
    # Preprocess: scale data so Rprop mathematics are stable
    print("Scaling 784-pixel data features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Standard MNIST split is 60,000 train, 10,000 test
    # Because there are 70,000 total, setting test_size=10000 gives us exactly 60000 for training.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)
    
    # Split the 60,000 training data mathematically among multiple clients
    # If num_clients=5, this gives exactly 12,000 images per client!
    client_X = np.array_split(X_train, num_clients)
    client_y = np.array_split(y_train, num_clients)
    
    clients_data = []
    for i in range(num_clients):
        clients_data.append((client_X[i], client_y[i]))
        print(f"Client {i+1} mathematically allocated {len(client_X[i])} high-res training samples.")
        
    global_test_data = (X_test, y_test)
    classes = np.unique(y) # Digits 0-9
    
    return clients_data, global_test_data, classes
