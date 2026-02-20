import pandas as pd
import os
import argparse

def load_data(output_path):
    # Raw data path setting
    # In a real scenario, this might come from an external source, but for this project
    # we'll create a copy of the raw dataset to the specified output path to simulate ingestion
    # or just load it from the source if it's already there
    
    source_path = os.path.join("data", "raw", "Churn Prediction DataSet.csv")
    
    # Check if file exists
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"File not found at {source_path}")
    
    df = pd.read_csv(source_path)
    print(f"Data Loaded Successfully. Shape: {df.shape}")
    
    # Save the data to the specified output path (ensuring directory exists)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/raw_data.csv", help="Path to save the raw data")
    args = parser.parse_args()
    
    load_data(args.output)