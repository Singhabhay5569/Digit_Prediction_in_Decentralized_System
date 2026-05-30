import hashlib

def hash_model_weights(weights, bias):
    """
    Converts the model weights and bias into a single SHA-256 hash.
    This creates a unique fingerprint for the model state.
    """
    if weights is None or bias is None:
        return hashlib.sha256(b"none").hexdigest()
        
    # Convert numpy arrays to bytes
    weights_bytes = b"".join(w.tobytes() for w in weights) if isinstance(weights, list) else weights.tobytes()
    bias_bytes = b"".join(b.tobytes() for b in bias) if isinstance(bias, list) else bias.tobytes()
    
    # Combine the bytes
    model_bytes = weights_bytes + bias_bytes
    
    # Generate SHA-256 hash
    hasher = hashlib.sha256()
    hasher.update(model_bytes)
    return hasher.hexdigest()
