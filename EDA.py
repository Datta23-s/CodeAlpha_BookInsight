import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv("books_data.csv")
print("✅ Data Loaded Successfully!\n")

# Step 2: Overview of data
print("🔍 Data Preview:")
print(df.head(), "\n")

print("📊 Data Info:")
print(df.info(), "\n")

print("🧮 Summary Statistics:")
print(df.describe(include='all'), "\n")

# Step 3: Data Cleaning
# Clean and convert price column (remove £ and Â if present)
df['Price'] = df['Price'].astype(str).str.replace('Â', '', regex=False).str.replace('£', '', regex=False).astype(float)

# Step 4: Questions & Explorations
# Q1. How many unique ratings are there?
print("🔢 Unique Ratings:")
print(df['Rating'].value_counts(), "\n")

# Q2. What is the average price of books by rating?
print("💰 Average Price by Rating:")
print(df.groupby('Rating')['Price'].mean(), "\n")

# Step 5: Visualizations
sns.set(style="whitegrid")

# Bar Plot: Count of books per rating
plt.figure(figsize=(8,5))
sns.countplot(x='Rating', data=df, order=df['Rating'].value_counts().index)
plt.title("Number of Books per Rating")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# Boxplot: Distribution of price by rating
plt.figure(figsize=(8,5))
sns.boxplot(x='Rating', y='Price', data=df)
plt.title("Book Price Distribution by Rating")
plt.show()

# Histogram: Price distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Price'], bins=10, kde=True)
plt.title("Price Distribution of Books")
plt.xlabel("Price (£)")
plt.ylabel("Count")
plt.show()

# Step 6: Anomaly detection
print("🔍 Books with Price > £50:")
print(df[df['Price'] > 50], "\n")
