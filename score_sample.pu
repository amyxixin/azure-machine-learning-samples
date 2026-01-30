"""
Simplified Batch Scoring Script for Azure ML
Only requires: input_data, output_data, model_path
Outputs direct model predictions without additional transformations.
"""
import os
import pandas as pd
import mlflow
import argparse


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True, help="Input data path")
    parser.add_argument("--output_data", type=str, required=True, help="Output data path")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    args = parser.parse_args()

    print(f"Input path: {args.input_data}")
    print(f"Output: {args.output_data}")
    print(f"Model path: {args.model_path}")

    # Load model
    model = mlflow.sklearn.load_model(args.model_path) # if model created with sklearn
    print("✅ Model loaded")

    # Get input files
    if os.path.isdir(args.input_data):
        files = [os.path.join(args.input_data, f) for f in os.listdir(args.input_data) if f.endswith(".csv")]
    else:
        files = [args.input_data]

    # Process each file
    results = []
    for file in files:
        print(f"📄 Processing: {file}")
        df = pd.read_csv(file)
        
        # Get predictions from model
        predictions = model.predict(df)
        
        # Add predictions to dataframe
        df["prediction"] = predictions
        
        results.append(df)

    # Save results
    if results:
        final_df = pd.concat(results, ignore_index=True)
        os.makedirs(args.output_data, exist_ok=True)
        output_file = os.path.join(args.output_data, "scored_results.csv")
        final_df.to_csv(output_file, index=False)
        print(f"✅ Results saved to: {output_file}")
    else:
        print("⚠️ No results to save")


if __name__ == "__main__":
    main()
