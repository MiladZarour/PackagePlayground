'''
you need to install first : 
pip install numpy
pip install pandas


This script covers the following topics:

Numpy: Creating arrays, performing mathematical operations, reshaping, and slicing.
Pandas: Creating DataFrames, displaying statistics, filtering, applying functions, merging, and exporting to CSV.
'''

import numpy as np
import pandas as pd

# Check versions
print("Using Numpy Version:", np.__version__)
print("Using Pandas Version:", pd.__version__)

# ---- NUMPY SECTION ----
print("\n=== NUMPY SECTION ===")

# Create a one-dimensional array
arr1 = np.array([1, 2, 3, 4, 5])
print("\n1D Array:")
print(arr1)

# Create a two-dimensional array
arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\n2D Array:")
print(arr2)

# Perform mathematical operations
print("\nSum of elements in arr1:", np.sum(arr1))
print("Mean of elements in arr2:", np.mean(arr2))

# Reshaping arrays
reshaped_arr2 = arr2.reshape(1, 9)
print("\nReshaped arr2:")
print(reshaped_arr2)

# Slicing arrays
print("\nSlicing arr2 to get the first two rows:")
print(arr2[:2])

# ---- PANDAS SECTION ----
print("\n=== PANDAS SECTION ===")

# Create a pandas DataFrame from a dictionary
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 22],
    'City': ['New York', 'Los Angeles', 'Chicago']
})
print("\nDataFrame:")
print(df)

# Display basic statistics
print("\nBasic statistics of DataFrame:")
print(df.describe(include='all'))

# Filtering data
filtered_df = df[df['Age'] > 25]
print("\nFilter people older than 25:")
print(filtered_df)

# Applying a function
df['Age'] = df['Age'].apply(lambda x: x + 1)
print("\nIncrement age by 1 using apply():")
print(df)

# Merging DataFrames
additional_data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Occupation': ['Engineer', 'Doctor', 'Writer']
})

merged_df = pd.merge(df, additional_data, on='Name')
print("\nMerged DataFrame with Occupation:")
print(merged_df)

# Exporting to CSV
merged_df.to_csv('output.csv', index=False)
print("\nExported the DataFrame to 'output.csv'")

print("\n=== END OF SCRIPT ===")
