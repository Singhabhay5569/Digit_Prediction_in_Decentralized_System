from data import load_and_split_data
from federated_rprop import IoTClient, FederatedServer
from blockchain import Blockchain
from utils import hash_model_weights
import matplotlib.pyplot as plt

def main():
    print("==================================================")
    print(" Secure Federated Learning with Blockchain Auth   ")
    print("==================================================")

    # 1. Setup Data - Splitting across 3 simulated IoT clients
    num_clients = 3
    clients_data, global_test_data, classes = load_and_split_data(num_clients)
    X_test, y_test = global_test_data

    # 2. Initialize the Blockchain system
    blockchain = Blockchain()
    print("\n[+] Blockchain initialized with Genesis block.")

    # 3. Initialize the Federated Server and the simulated Edge IoT Clients
    server = FederatedServer()
    clients = []
    for i in range(num_clients):
        X_client, y_client = clients_data[i]
        client = IoTClient(f"Client_{i+1}", X_client, y_client, classes)
        clients.append(client)
    
    print(f"\n[+] {num_clients} IoT Edge Clients simulated and ready (Using Rprop).")

    # 4. Federated Training Loop
    num_rounds = 5
    
    # Initialize live plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Live Convergence Graph (Rprop Federated Learning)")
    ax.set_xlabel("Federated Rounds")
    ax.set_ylabel("Global Accuracy (%)")
    ax.set_xlim(0, num_rounds + 1)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    line, = ax.plot([], [], marker='o', color='b', linewidth=2, markersize=8)
    
    round_history = []
    acc_history = []
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Federated Round {round_num} ---")
        
        client_params = []
        
        # A. Train each client locally (Simulating decentralized processing)
        for i, client in enumerate(clients):
            # Send the current global model state to client (skip first round)
            client.set_weights(server.global_weights, server.global_bias)
            
            # Client trains locally strictly on their private dataset slice
            w, b = client.train()
            client_params.append((w, b))
            print(f"   [IoT {client.client_id}] Local Model trained successfully.")
            
        # B. Central Server securely aggregates the parameter matrix blocks
        print("   [Server] Aggregating local models using FedAvg...")
        global_w, global_b = server.aggregate_models(client_params)
        
        # C. Security Implementation - Cryptographically hashing the collective global state
        model_hash = hash_model_weights(global_w, global_b)
        print(f"   [Security] Iterational Model Hash (SHA-256): {model_hash[:16]}...")
        
        # D. Store iteration directly onto our simulated local blockchain ledger
        block_data = f"Round: {round_num}, Model_Hash: {model_hash}"
        blockchain.add_block(block_data)
        print("   [Blockchain] Model update recorded securely.")
        
        # E. Final Performance Evaluation
        acc = server.evaluate_global_model(X_test, y_test, classes)
        print(f"   [Evaluation] Global Model Current Accuracy: {acc * 100:.2f}%")
        
        # Check standard blockchain chain mathematical validity ensuring trust
        is_valid = blockchain.is_chain_valid()
        print(f"   [Blockchain Status] Cryptographically Valid: {is_valid}")
        
        # Update live convergence graph dynamically
        round_history.append(round_num)
        acc_history.append(acc * 100)
        line.set_xdata(round_history)
        line.set_ydata(acc_history)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.5)  # Pause to make the real-time animation visible

    # 5. Display the entire valid finalized blockchain ledger properly
    blockchain.print_chain()

    # 6. Security Demonstration - Tampering Vector Attack Simulation 
    print("\n==================================================")
    print(" SECURITY DEMONSTRATION: Simulated Tampering Attack ")
    print("==================================================")
    print("   [Hacker] Attempting to maliciously modify block #2 data...")
    
    # Tampering with block 2 (e.g. malicious party trying to claim a different trained state in the past)
    blockchain.chain[2].data = "Round: 2, Model_Hash: HACKED_MALICIOUS_HASH"
    
    print("   [System] Verifying collective blockchain integrity...")
    is_valid_after_attack = blockchain.is_chain_valid()
    
    if not is_valid_after_attack:
        print("   [ALERT] Blockchain validation FAILED! Tampering detected via invalid hash linkages.")
    else:
        print("   [?] Blockchain validation passed (This logically shouldn't happen!).")
        
    print("==================================================\n")
    
    print("[*] Training Complete! Close the graph window to exit the program.")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
