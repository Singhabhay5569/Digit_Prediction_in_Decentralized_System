# Digit_Prediction_in_Decentralized_System
Developed a privacy-preserving Federated Learning framework for distributed IoT devices using the MNIST dataset. Implemented a SHA-256 blockchain ledger to validate model updates and detect tampering. Optimized training with Rprop and evaluated performance using accuracy, Convergence Graph.
# Secure Federated Learning with Simulated Blockchain Validation for Edge IoT Systems

## Project Objective
This project functionally simulates a secure edge learning environment where multiple IoT (Internet of Things) devices collaboratively train a unified machine learning model without directly sharing sensitive underlying raw data. Instead of sharing raw private device data, only processed parameter weights are shared centrally. To guarantee transparency, security, and un-tampered authenticity of these parameters, a simulated blockchain framework creates cryptographic SHA-256 links logging the validated model's evolutionary updates over sequential training rounds. It actively thwarts historical ledger tampering as demonstrated seamlessly within the executed lifecycle.

## Technologies Used
- **Python 3**: Core execution programming language.
- **NumPy**: Employed for fundamental linear algebra and matrix-level Federated Averaging mathematically.
- **scikit-learn**: Primarily manages the machine learning processing (`SGDClassifier` scaling local predictive Logistic Regression functions incrementally) and the structured importing pipeline for the benchmark dataset (MNIST Digits array).
- **hashlib**: Built-in library leveraged entirely for SHA-256 secure mathematical hash calculations bridging block data segments together.

## Execution and Installation: Step-By-Step
1. **Ensure System Requirements**: Make certain you have standard `Python 3` installed on your machine.
2. **Install Core Dependencies**:
   Open up your terminal or designated command prompt. Execute the standard pip command to get the required statistical processing packages strictly:
   ```bash
   pip install numpy scikit-learn
   ```
3. **Execute Logic Flow Application**:
   Navigate into the project directory and seamlessly run the orchestrator script:
   ```bash
   python main.py
   ```

## Expected Visual Output Logging
Upon running the central executable script seamlessly, you will visibly observe the real-time procedural logging:
1. Core pipeline setup, initializing the exact dataset and simulated distribution slices across the IoT Clients.
2. The genesis formation of the pristine root node blockchain sequence.
3. Looped multi-round simulated IoT Edge collaborative processing seamlessly implementing mathematical Federated Averaging (scaling global accuracy directly per iteration).
4. Active Blockchain monitoring securely appending cryptographic timestamps validating the global model update.
5. Absolute state snapshot visualization outputting standard readable matrix block attributes safely.
6. A programmatic internal validation functionally demonstrating simulated network tampering detection explicitly where deliberately altered text seamlessly snaps linkage structures logically alerting system administrators.
