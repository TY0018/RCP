import pandas as pd

# 1. Load your CSV
df = pd.read_csv("/home/users/ntu/ytong005/scratch/asian_bird_dataset/Asian_Birds/xc_metadata.csv")

# 2. Count and Sort
# value_counts() returns a Series where index is the category and value is the count
category_counts = df['categories'].value_counts()

# 3. Print the result
print(category_counts)
