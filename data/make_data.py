import pandas as pd
import numpy as np

# Create date range (60 months starting from January 2000)
dates = pd.date_range(start='2000-01-01', periods=60, freq='MS')
df = pd.DataFrame({'date': dates})

# Map months to seasons
month_to_season = {
    1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall',
    12: 'Winter'
}
df['season'] = df['date'].dt.month.map(month_to_season)

# Create volume with seasonal patterns
spring_fall_mask = df['season'].isin(['Spring', 'Fall'])
df.loc[spring_fall_mask, 'volume'] = np.random.randint(300000, 500001, size=spring_fall_mask.sum())
df.loc[~spring_fall_mask, 'volume'] = np.random.randint(100000, 300001, size=(~spring_fall_mask).sum())

# Create avg_flag based on volume
df['avg_flag'] = (df['volume'] > 300000).astype(int)

# Create clarity column with 5% missing values
clarity_options = ['Very high', 'High', 'Medium', 'Low', 'Very low']
df['clarity'] = np.random.choice(clarity_options, size=60)
missing_indices = np.random.choice(60, size=int(60 * 0.05), replace=False)
df.loc[missing_indices, 'clarity'] = np.nan

# Create samples column with 10% missing values
df['samples'] = np.random.randint(1, 11, size=60)
missing_indices = np.random.choice(60, size=int(60 * 0.10), replace=False)
df.loc[missing_indices, 'samples'] = np.nan

# Create traffic column with 10% missing values
df['traffic'] = np.random.randint(0, 1001, size=60)
missing_indices = np.random.choice(60, size=int(60 * 0.10), replace=False)
df.loc[missing_indices, 'traffic'] = np.nan

# Print-check for data
print(df)

# Export to CSV
df.to_csv('C:/Users/doncs/Documents/GitHub/TADPREP/data/river_data.csv', index=False)
