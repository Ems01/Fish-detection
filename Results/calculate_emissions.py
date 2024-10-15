import pandas as pd

# Load the emissions CSV file
emissions_file = 'Results/emissions.csv'
df = pd.read_csv(emissions_file)

# Strip extra spaces from column names
df.columns = df.columns.str.strip()

# Calculate total emissions
total_emissions = df['emissions'].sum()

print(f"Total Emissions: {total_emissions:.2f} kg CO2")
