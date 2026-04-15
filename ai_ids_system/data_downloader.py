import pandas as pd
import numpy as np
import os

DATA_FILE = "nsl_kdd_synthetic.csv"

def generate_synthetic_data(num_samples=1000):
    """
    Since downloading NSL-KDD automatically requires authentication or specific mirrors,
    we generate a synthetic dataset that mimics the structure of NSL-KDD for training purposes.
    Replace this with the actual NSL-KDD CSV file when deploying in production.
    """
    print(f"[*] Generating {num_samples} samples of synthetic NSL-KDD data...")
    
    # Typical NSL-KDD features
    np.random.seed(42)
    data = {
        'duration': np.random.exponential(scale=10, size=num_samples),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], size=num_samples),
        'service': np.random.choice(['http', 'ftp', 'smtp', 'domain_u'], size=num_samples),
        'flag': np.random.choice(['SF', 'S0', 'REJ'], size=num_samples),
        'src_bytes': np.random.exponential(scale=500, size=num_samples),
        'dst_bytes': np.random.exponential(scale=1000, size=num_samples),
        'count': np.random.randint(1, 100, size=num_samples),
        'srv_count': np.random.randint(1, 100, size=num_samples),
        # Target variable: 'normal' or various attack types like 'neptune', 'smurf', 'satan'
        'label': np.random.choice(['normal', 'neptune', 'smurf', 'satan', 'ipsweep'], size=num_samples, p=[0.5, 0.2, 0.1, 0.1, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Binary classification logic: normal -> 0, attack -> 1
    df['attack_class'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    df.to_csv(DATA_FILE, index=False)
    print(f"[+] Synthetic data saved to {DATA_FILE}")

if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        generate_synthetic_data()
    else:
        print(f"[*] {DATA_FILE} already exists. Ready for training.")
