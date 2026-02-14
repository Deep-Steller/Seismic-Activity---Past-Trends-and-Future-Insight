import pandas as pd
import os

# Load the dataset
file_path = os.path.join('data', 'Significant_Earthquakes.csv')
df = pd.read_csv(file_path)

# ------------------------------------------------------
# 1. Initial Exploration
# ------------------------------------------------------

# Show 5 random sample rows
print("Random Samples:")
print(df.sample(5))

# Show data types
print("\nData Types:")
print(df.dtypes)

# Basic statistics of numeric columns
print("\nBasic Statistics:")
print(df.describe())

# Missing value summary
print("\nMissing Values Summary:")
print(df.isnull().sum())

# Top 10 most common places (if 'place' column exists)
if 'place' in df.columns:
    print("\nTop 10 Places:")
    print(df['place'].value_counts().head(10))

# Count of different event types
if 'type' in df.columns:
    print("\nEarthquake Types:")
    print(df['type'].value_counts())

# ------------------------------------------------------
# 2. Drop Columns with High Missing Values or Low Relevance
# ------------------------------------------------------
# Reasoning for dropping: Mostly missing and won't be used for prediction
# - 'nst', 'gap', 'dmin', 'rms': Seismological metadata, 'horizontalError', 'depthError', 'magError', 'magNst'
columns_to_drop = [
    'nst',              # Number of stations used
    'gap',              # Azimuthal gap
    'dmin',             # Distance to nearest station
    'rms',              # Root mean square of amplitude
    'horizontalError',  # Horizontal error in km
    'depthError',       # Uncertainty in depth
    'magError',         # Uncertainty in magnitude
    'magNst'            # Number of stations for mag
]

# Drop if columns exist in dataset
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# ------------------------------------------------------
# 3. Clean Important Fields
# ------------------------------------------------------

# Convert 'time' to datetime format
df['time'] = pd.to_datetime(df['time'], errors='coerce')

# Drop rows with missing key fields
df.dropna(subset=['time', 'latitude', 'longitude', 'mag'], inplace=True)

# Fill missing depth with median 
if 'depth' in df.columns:
    df['depth'] = df['depth'].fillna(df['depth'].median())

# Fill missing 'place' with 'Unknown'
if 'place' in df.columns:
    df['place'] = df['place'].fillna('Unknown')

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Normalize column names for consistency
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# ------------------------------------------------------
# 4. Save Cleaned Dataset
# ------------------------------------------------------
df.to_csv('data/cleaned_earthquakes.csv', index=False)

# Final summary
print("\nFinal Columns:", df.columns.tolist())
print("Final Shape:", df.shape)
print("Missing Summary:\n", df.isnull().sum())
print("\nCleaning completed. Cleaned data saved to 'data/cleaned_earthquakes.csv'.")
