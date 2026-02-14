import pandas as pd
import mysql.connector

# Load the cleaned data
df = pd.read_csv('data/cleaned_earthquakes.csv')

# Convert time to MySQL-compatible string format
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Replace NaN with None for MySQL NULL insertion
df = df.where(pd.notnull(df), None)

# Connect to MySQL
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='1q2w3e4r5',
    database='seismic_data'
)
cursor = conn.cursor()

# Insert each row
for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO earthquakes (
            time, latitude, longitude, depth, mag, magtype, net,
            place, type, status, locationsource, magsource
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        row['time'], row['latitude'], row['longitude'], row['depth'],
        row['mag'], row['magtype'], row['net'], row['place'], row['type'],
        row['status'], row['locationsource'], row['magsource']
    ))

# Finalize
conn.commit()
cursor.close()
conn.close()

print("Data successfully inserted into MySQL!")
