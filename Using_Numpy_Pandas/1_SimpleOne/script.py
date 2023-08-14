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



'''
OUTPUT>>>>>>>>>>>>
Using Numpy Version: 1.25.2
Using Pandas Version: 2.0.3

=== NUMPY SECTION ===

1D Array:
[1 2 3 4 5]

2D Array:
[[1 2 3]
 [4 5 6]
 [7 8 9]]

Sum of elements in arr1: 15
Mean of elements in arr2: 5.0

Reshaped arr2:
[[1 2 3 4 5 6 7 8 9]]

Slicing arr2 to get the first two rows:
[[1 2 3]
 [4 5 6]]

=== PANDAS SECTION ===

DataFrame:
      Name  Age         City
0    Alice   25     New York
1      Bob   30  Los Angeles
2  Charlie   22      Chicago

Basic statistics of DataFrame:
         Name        Age      City
count       3   3.000000         3
unique      3        NaN         3
top     Alice        NaN  New York
freq        1        NaN         1
mean      NaN  25.666667       NaN
std       NaN   4.041452       NaN
min       NaN  22.000000       NaN
25%       NaN  23.500000       NaN
50%       NaN  25.000000       NaN
75%       NaN  27.500000       NaN
max       NaN  30.000000       NaN

Filter people older than 25:
  Name  Age         City
1  Bob   30  Los Angeles

Increment age by 1 using apply():
      Name  Age         City
0    Alice   26     New York
1      Bob   31  Los Angeles
2  Charlie   23      Chicago

Merged DataFrame with Occupation:
      Name  Age         City Occupation
0    Alice   26     New York   Engineer
1      Bob   31  Los Angeles     Doctor
2  Charlie   23      Chicago     Writer

Exported the DataFrame to 'output.csv'

=== END OF SCRIPT ===
'''
