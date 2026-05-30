import numpy as np
from sklearn.metrics import accuracy_score

def softmax(z):
    # Numerically stable softmax
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class RpropClassifier:
    """
    Custom implementation of Resilient Propagation (Rprop) using NumPy.
    Implements a 1-hidden-layer Neural Network with a 'tanh' activation function.
    """
    def __init__(self, hidden_layer_size=32, random_state=42):
        self.hidden_layer_size = hidden_layer_size
        self.random_state = random_state
        np.random.seed(random_state)
        self.coefs_ = None
        self.intercepts_ = None
        self.classes = None
        self._rprop_initialized = False
        
    def partial_fit(self, X, y, classes=None):
        if self.classes is None:
            self.classes = classes
            
        num_classes = len(self.classes)
        num_features = X.shape[1]
        
        if self.coefs_ is None:
            # Initialize weights randomly for 2 layers
            # Layer 1: Input -> Hidden
            W1 = np.random.randn(num_features, self.hidden_layer_size) * np.sqrt(2. / num_features) # He/Xavier initialization approximation
            b1 = np.zeros(self.hidden_layer_size)
            # Layer 2: Hidden -> Output
            W2 = np.random.randn(self.hidden_layer_size, num_classes) * np.sqrt(2. / self.hidden_layer_size)
            b2 = np.zeros(num_classes)
            
            self.coefs_ = [W1, W2]
            self.intercepts_ = [b1, b2]
            
        if not self._rprop_initialized:
            self.rprop_state = {
                'step_size_w': [np.full_like(self.coefs_[0], 0.1), np.full_like(self.coefs_[1], 0.1)],
                'step_size_b': [np.full_like(self.intercepts_[0], 0.1), np.full_like(self.intercepts_[1], 0.1)],
                'prev_grad_w': [np.zeros_like(self.coefs_[0]), np.zeros_like(self.coefs_[1])],
                'prev_grad_b': [np.zeros_like(self.intercepts_[0]), np.zeros_like(self.intercepts_[1])],
                'prev_update_w': [np.zeros_like(self.coefs_[0]), np.zeros_like(self.coefs_[1])],
                'prev_update_b': [np.zeros_like(self.intercepts_[0]), np.zeros_like(self.intercepts_[1])]
            }
            self._rprop_initialized = True
            
        # One-hot encode y
        Y = np.zeros((X.shape[0], num_classes))
        for i, val in enumerate(y):
            idx = np.where(self.classes == val)[0][0]
            Y[i, idx] = 1.0
            
        # ---------------- FORWARD PASS ----------------
        W1, W2 = self.coefs_
        b1, b2 = self.intercepts_
        
        # Layer 1 (Hidden Layer - Tanh)
        Z1 = np.dot(X, W1) + b1
        A1 = np.tanh(Z1)
        
        # Layer 2 (Output Layer - Softmax)
        Z2 = np.dot(A1, W2) + b2
        A2 = softmax(Z2)
        
        # ---------------- BACKPROPAGATION ----------------
        m = X.shape[0]
        
        # Output layer gradients
        dZ2 = A2 - Y
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0) / m
        
        # Hidden layer gradients
        dA1 = np.dot(dZ2, W2.T)
        # Derivative of tanh(Z) is 1 - tanh^2(Z) == 1 - A^2
        dZ1 = dA1 * (1.0 - np.power(A1, 2))
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0) / m
        
        # Apply strict Rprop optimization rules mapped per layer
        self._rprop_update('w', 0, dW1)
        self._rprop_update('w', 1, dW2)
        self._rprop_update('b', 0, db1)
        self._rprop_update('b', 1, db2)
        
    def _rprop_update(self, param_type, layer_idx, grad):
        state = self.rprop_state
        step_sz = state[f'step_size_{param_type}'][layer_idx]
        prev_grad = state[f'prev_grad_{param_type}'][layer_idx]
        prev_update = state[f'prev_update_{param_type}'][layer_idx]
        
        param = self.coefs_[layer_idx] if param_type == 'w' else self.intercepts_[layer_idx]
        
        sign_prod = grad * prev_grad
        
        # 1. sign_prod > 0 (gradient retains sign -> accelerate)
        pos_mask = sign_prod > 0
        step_sz[pos_mask] = np.minimum(step_sz[pos_mask] * 1.2, 50.0)
        update_pos = -np.sign(grad) * step_sz
        param[pos_mask] += update_pos[pos_mask]
        prev_update[pos_mask] = update_pos[pos_mask]
        prev_grad[pos_mask] = grad[pos_mask]
        
        # 2. sign_prod < 0 (gradient reversed sign -> went too far -> backtrack)
        neg_mask = sign_prod < 0
        step_sz[neg_mask] = np.maximum(step_sz[neg_mask] * 0.5, 1e-6)
        param[neg_mask] -= prev_update[neg_mask]
        prev_grad[neg_mask] = 0 # Force next step to be 0 for this weight
        
        # 3. sign_prod == 0
        zero_mask = sign_prod == 0
        update_zero = -np.sign(grad) * step_sz
        param[zero_mask] += update_zero[zero_mask]
        prev_update[zero_mask] = update_zero[zero_mask]
        prev_grad[zero_mask] = grad[zero_mask]
        
    def predict(self, X):
        W1, W2 = self.coefs_
        b1, b2 = self.intercepts_
        
        Z1 = np.dot(X, W1) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = softmax(Z2)
        return self.classes[np.argmax(A2, axis=1)]


class IoTClient:
    """
    Simulates an IoT device that trains a machine learning model locally 
    on its own private dataset slice using Resilient Propagation (Rprop).
    """
    def __init__(self, client_id, X, y, classes):
        self.client_id = client_id
        self.X = X
        self.y = y
        self.classes = classes
        
        # Utilizing Custom Rprop implementation with a Tanh Hidden Layer of size 32
        self.local_model = RpropClassifier(hidden_layer_size=32, random_state=42)
        
        # Initialize the model weights to the proper dimensions
        self.local_model.partial_fit(self.X[:1], self.y[:1], classes=self.classes)
        
    def set_weights(self, global_weights, global_bias):
        """ Overwrite local model parameters with globally aggregated parameters. """
        if global_weights is not None and global_bias is not None:
            self.local_model.coefs_ = [np.copy(w) for w in global_weights]
            self.local_model.intercepts_ = [np.copy(b) for b in global_bias]

    def train(self):
        """ Train the local model and return the updated weights & bias. """
        # Simulate local training rounds via Rprop
        for _ in range(5):
            self.local_model.partial_fit(self.X, self.y)
        
        return self.local_model.coefs_, self.local_model.intercepts_


class FederatedServer:
    """
    Simulates the central server that aggregates models from all IoT clients 
    WITHOUT ever seeing their raw data.
    """
    def __init__(self):
        self.global_weights = None
        self.global_bias = None
        
    def aggregate_models(self, client_parameters):
        """
        Federated Averaging (FedAvg): average the weights and biases from all clients.
        """
        num_clients = len(client_parameters)
        if num_clients == 0:
            return None, None
            
        num_layers = len(client_parameters[0][0])
        sum_weights = [np.zeros_like(client_parameters[0][0][i]) for i in range(num_layers)]
        sum_bias = [np.zeros_like(client_parameters[0][1][i]) for i in range(num_layers)]
        
        for w, b in client_parameters:
            for i in range(num_layers):
                sum_weights[i] += w[i]
                sum_bias[i] += b[i]
            
        self.global_weights = [sw / num_clients for sw in sum_weights]
        self.global_bias = [sb / num_clients for sb in sum_bias]
        
        return self.global_weights, self.global_bias

    def evaluate_global_model(self, X_test, y_test, classes):
        """ Evaluate the current global model accuracy on a holdout test set. """
        if self.global_weights is None:
            return 0.0
            
        test_model = RpropClassifier(hidden_layer_size=32, random_state=42)
        test_model.partial_fit(X_test[:1], y_test[:1], classes=classes)
        test_model.coefs_ = [np.copy(w) for w in self.global_weights]
        test_model.intercepts_ = [np.copy(b) for b in self.global_bias]
        
        predictions = test_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy
