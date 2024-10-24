import os
import pandas as pd

# Path to the directory containing subfolders with emissions files
directory_path = 'Results'
output_file = 'Results\emissions_summary.txt'

# Initialize a dictionary to hold the sum of emissions for each subfolder
emissions_sums = {}

# Print the initial directory structure for debugging
print("Checking directory structure:")
for subdir, _, files in os.walk(directory_path):
    print(f"Found directory: {subdir}")  # Debug print
    for file in files:
        print(f"Found file: {file}")  # Debug print

# Loop through each subfolder in the directory
for subdir, _, files in os.walk(directory_path):
    for file in files:
        if file == 'emissions.csv':  # Check if the file is named 'emissions.csv'
            file_path = os.path.join(subdir, file)
            print(f"Processing file: {file_path}")  # Debug print

            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Strip extra spaces from column names
            df.columns = df.columns.str.strip()

            # Check if the 'emissions' column exists
            if 'emissions' in df.columns:
                print(f"Found 'emissions' column in {file_path}")  # Debug print
                
                # Print the contents of the emissions column for debugging
                print(f"Emissions data:\n{df['emissions']}")  # Debug print

                # Attempt to convert emissions to numeric and handle errors
                df['emissions'] = pd.to_numeric(df['emissions'], errors='coerce')  # Convert to numeric

                # Check for NaN values
                nan_count = df['emissions'].isna().sum()
                print(f"Number of NaN values in 'emissions': {nan_count}")  # Debug print
                
                # Calculate total emissions, ignoring NaN values
                total_emissions = df['emissions'].sum()
                print(f"Total emissions from {file_path}: {total_emissions:.2f} kg CO2")  # Debug print

                subfolder_name = os.path.basename(subdir)
                emissions_sums[subfolder_name] = emissions_sums.get(subfolder_name, 0) + total_emissions
            else:
                print(f"'emissions' column not found in {file_path}")  # Debug print

# Calculate the grand total emissions
grand_total_emissions = sum(emissions_sums.values())

# Write the results to a text file
with open(output_file, 'w') as f:
    for subfolder, total in emissions_sums.items():
        f.write(f"{subfolder}: {total:.2f} kg CO2\n")
    f.write(f"\nTotal Emissions: {grand_total_emissions:.2f} kg CO2\n")

print(f"Results saved to {output_file}")
