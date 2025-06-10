import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and clean data
df = pd.read_csv("books_data.csv")
df['Price'] = df['Price'].astype(str).str.replace('Â', '', regex=False).str.replace('£', '', regex=False).astype(float)

# Step 2: Set visualization theme
sns.set(style="whitegrid")
plt.rcParams.update({'figure.figsize': (8, 5), 'figure.dpi': 100})

# Step 3: Bar plot – Book count by rating
plt.figure()
sns.countplot(x='Rating', data=df, order=df['Rating'].value_counts().index, palette="pastel")
plt.title("📚 Number of Books by Rating")
plt.xlabel("Rating")
plt.ylabel("Number of Books")
plt.tight_layout()
plt.show()

# Step 4: Box plot – Price distribution by rating
plt.figure()
sns.boxplot(x='Rating', y='Price', data=df, palette="coolwarm")
plt.title("💰 Book Price Distribution by Rating")
plt.xlabel("Rating")
plt.ylabel("Price (£)")
plt.tight_layout()
plt.show()

# Step 5: Histogram – Price distribution
plt.figure()
sns.histplot(df['Price'], bins=15, kde=True, color='skyblue')
plt.title("📈 Price Distribution of Books")
plt.xlabel("Price (£)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Step 6: Pie chart – Rating proportions
rating_counts = df['Rating'].value_counts()
plt.figure()
plt.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title("🧁 Distribution of Book Ratings")
plt.tight_layout()
plt.show()

# Step 7: Scatter Plot – Price vs Rating (encoded)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Rating_Num'] = le.fit_transform(df['Rating'])

plt.figure()
sns.scatterplot(x='Rating_Num', y='Price', data=df, hue='Rating', palette='deep', alpha=0.7)
plt.title("🔍 Book Price vs Encoded Rating")
plt.xlabel("Encoded Rating")
plt.ylabel("Price (£)")
plt.tight_layout()
plt.show()
