import hashlib
import time

class Block:
    """
    Represents a single block in the blockchain holding the aggregated ML model hash.
    """
    def __init__(self, index, data, previous_hash):
        self.index = index
        self.timestamp = time.time()
        self.data = data  # Contains round number and model hash
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        """ Calculate the SHA-256 hash of the block itself """
        block_string = f"{self.index}{self.timestamp}{self.data}{self.previous_hash}"
        return hashlib.sha256(block_string.encode('utf-8')).hexdigest()
        
    def __str__(self):
        return (f"Block #{self.index} |\n"
                f"Data: {self.data} |\n"
                f"Prev Hash: {self.previous_hash[:10]}... |\n"
                f"Hash: {self.hash[:10]}...\n")


class Blockchain:
    """
    A simulated blockchain to record and validate federated learning model updates.
    """
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        """ Creates the very first block in the blockchain completely hardcoded. """
        return Block(0, "Genesis Block - System Init", "0")

    def get_latest_block(self):
        """ Returns the last block in the chain """
        return self.chain[-1]

    def add_block(self, new_data):
        """
        Creates a new block with updated data and the previous block's hash,
        then securely appends it to the chain.
        """
        previous_block = self.get_latest_block()
        new_block = Block(
            index=previous_block.index + 1,
            data=new_data,
            previous_hash=previous_block.hash
        )
        self.chain.append(new_block)
        return new_block

    def is_chain_valid(self):
        """
        Validates the blockchain by checking:
        1. Whether the newly calculated hash of each block is correct.
        2. Whether each block logically points to the correct previous hash.
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            # Re-calculate hash to see if the block content was tampered with
            if current_block.hash != current_block.calculate_hash():
                return False

            # Check if the block is linked to the correct previous block
            if current_block.previous_hash != previous_block.hash:
                return False

        return True

    def print_chain(self):
        print("\n--- Current Blockchain State ---")
        for block in self.chain:
            print(block)
        print("--------------------------------")
