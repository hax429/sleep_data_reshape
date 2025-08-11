import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Read the CSV file
df = pd.read_csv('/example/eda_left_data.csv')

# Convert timestamps to datetime
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], format='ISO8601')
df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])

# Basic statistics
print("=== EDA Data Summary ===")
print(f"Dataset shape: {df.shape}")
print(f"Time range: {df['timestamp_local'].min()} to {df['timestamp_local'].max()}")
print(f"Duration: {df['timestamp_local'].max() - df['timestamp_local'].min()}")
print(f"Sample rate: ~{1/(df['timestamp_local'].diff().dt.total_seconds().mode()[0])} Hz")
print()

print("=== EDA Signal Statistics ===")
print(f"Mean EDA: {df['eda'].mean():.6f}")
print(f"Std EDA: {df['eda'].std():.6f}")
print(f"Min EDA: {df['eda'].min():.6f}")
print(f"Max EDA: {df['eda'].max():.6f}")
print(f"Range: {df['eda'].max() - df['eda'].min():.6f}")
print()

print("=== Other Variables ===")
print(f"Timezone: {df['timezone'].unique()}")
print(f"Day of study: {df['day_of_study'].unique()}")
print(f"Device worn: {df['device_worn'].unique()}")
print()

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Time series plot
axes[0, 0].plot(df['timestamp_local'], df['eda'], linewidth=0.8, alpha=0.8)
axes[0, 0].set_title('EDA Signal Over Time')
axes[0, 0].set_xlabel('Local Time')
axes[0, 0].set_ylabel('EDA (µS)')
axes[0, 0].tick_params(axis='x', rotation=45)

# Histogram
axes[0, 1].hist(df['eda'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 1].set_title('EDA Value Distribution')
axes[0, 1].set_xlabel('EDA (µS)')
axes[0, 1].set_ylabel('Frequency')

# Box plot
axes[1, 0].boxplot(df['eda'])
axes[1, 0].set_title('EDA Box Plot')
axes[1, 0].set_ylabel('EDA (µS)')
axes[1, 0].set_xticklabels(['EDA'])

# Rolling statistics
window = 20  # 20 sample rolling window (~5 seconds at 4Hz)
df['eda_rolling_mean'] = df['eda'].rolling(window=window, center=True).mean()
df['eda_rolling_std'] = df['eda'].rolling(window=window, center=True).std()

axes[1, 1].plot(df['timestamp_local'], df['eda'], alpha=0.3, label='Raw EDA', linewidth=0.5)
axes[1, 1].plot(df['timestamp_local'], df['eda_rolling_mean'], label=f'{window}-point Moving Average', linewidth=1.5)
axes[1, 1].fill_between(df['timestamp_local'], 
                       df['eda_rolling_mean'] - df['eda_rolling_std'], 
                       df['eda_rolling_mean'] + df['eda_rolling_std'], 
                       alpha=0.2, label='±1 SD')
axes[1, 1].set_title('EDA with Moving Average')
axes[1, 1].set_xlabel('Local Time')
axes[1, 1].set_ylabel('EDA (µS)')
axes[1, 1].legend()
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/Users/hax429/Developer/Internship/reshape/eda_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional analysis
print("=== Signal Quality Analysis ===")
# Look for potential artifacts or unusual patterns
high_variation = df['eda'].diff().abs() > 2 * df['eda'].diff().abs().std()
print(f"High variation points: {high_variation.sum()} ({high_variation.mean()*100:.2f}%)")

# Check for flat periods (could indicate sensor issues)
flat_periods = df['eda'].diff().abs() < 0.001
consecutive_flat = (flat_periods & flat_periods.shift(1) & flat_periods.shift(-1))
print(f"Potential flat periods: {consecutive_flat.sum()} samples")

print("=== Temporal Analysis ===")
print(f"Start time: {df['timestamp_local'].iloc[0]}")
print(f"End time: {df['timestamp_local'].iloc[-1]}")
print(f"Total duration: {df['timestamp_local'].iloc[-1] - df['timestamp_local'].iloc[0]}")
print(f"Total samples: {len(df)}")