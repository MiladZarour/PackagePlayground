'''
you need to install first : 
pip install numpy
pip install pandas

This script illustrates:

-Numpy:
1- Creation of random matrices
2- Eigenvalue decomposition
3- Singular Value Decomposition (SVD)
4- Fourier Transform

Pandas:
1- Creation of a complex DataFrame including date operations
2- Resampling and group by operations
3- Pivot tables
4- Rolling window operations
5- Joining DataFrames
6- Normalizing data using the apply() method
'''


import numpy as np
import pandas as pd

print("Using Numpy Version:", np.__version__)
print("Using Pandas Version:", pd.__version__)

# ---- NUMPY SECTION ----
print("\n=== NUMPY SECTION ===")

# Creating random arrays
random_matrix = np.random.rand(5, 5)
print("\nRandom 5x5 Matrix:")
print(random_matrix)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(random_matrix)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)

# SVD decomposition
U, S, VT = np.linalg.svd(random_matrix)
print("\nSVD Decomposition:")
print("U:")
print(U)
print("S:")
print(S)
print("VT:")
print(VT)

# Fourier Transform
fourier_transform = np.fft.fft(np.sin(np.linspace(0, 2 * np.pi, 100)))
print("\nFourier Transform:")
print(fourier_transform)

# ---- PANDAS SECTION ----
print("\n=== PANDAS SECTION ===")

# Create a complex DataFrame
date_rng = pd.date_range(start='2020-01-01', end='2020-12-31', freq='M')
df = pd.DataFrame(date_rng, columns=['date'])
df['data'] = np.random.randint(0, 100, size=(len(date_rng)))
df['category'] = ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D']
print("\nComplex DataFrame:")
print(df)

# Setting the date column as index
df.set_index('date', inplace=True)

# Resampling the data
monthly_summary = df.resample('Q').mean()
print("\nQuarterly Summary:")
print(monthly_summary)

# Group by operations
grouped = df.groupby(df.index.month)['data'].agg(['mean', 'sum', 'max'])
print("\nMonthly Group Summary:")
print(grouped)

# Pivot Table
pivot_table = pd.pivot_table(df, index=df.index.month, columns='category', values='data', aggfunc='sum')
print("\nPivot Table:")
print(pivot_table)

# Rolling window operations
rolling_mean = df['data'].rolling(window=3).mean()
print("\n3-Month Rolling Mean:")
print(rolling_mean)

# Joining DataFrames
additional_data = pd.DataFrame({
    'month': range(1, 13),
    'factor': np.random.rand(12)
})
additional_data.set_index('month', inplace=True)
joined_df = grouped.join(additional_data, on='mean')
print("\nJoined DataFrame with additional data:")
print(joined_df)

# Apply function to normalize data
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

normalized_df = df['data'].apply(normalize)
print("\nNormalized Data:")
print(normalized_df)

print("\n=== END OF SCRIPT ===")
