import os
import pandas as pd
import numpy as np
# Specify the folder containing your CSV files
dataset = ["itunes-amazon-dirty", "abt-buy-textual", "dblp-scholar-dirty", "dblp-scholar-structured"]
folder_list = []
for dataset in dataset:
    folder_list.append('./sample_data/' + dataset + '/learn_data/')

# Loop through all files in the folder
for folder_path in folder_list:
    print("path:", folder_path)
    for filename in os.listdir(folder_path):
        # Check if the file is a CSV file
        if filename.endswith('.csv'):
            # Construct the full path to the CSV file
            file_path = os.path.join(folder_path, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            # Replace NaN values with an empty string
            df = df.replace(np.nan, "None")
            # Save the updated DataFrame back to the CSV file
            df.to_csv(file_path, index=False)
            print(f"Filled NaN values in {filename}")

print("Processing complete.")