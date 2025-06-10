import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import random

# Download VADER Lexicon
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

# Step 1: Scrape book titles from Books to Scrape
def scrape_book_titles(pages=2):
    base_url = "http://books.toscrape.com/catalogue/page-{}.html"
    titles = []

    for page in range(1, pages + 1):
        url = base_url.format(page)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        items = soup.select("article.product_pod h3 a")
        for item in items:
            titles.append(item['title'])
    
    return titles

# Step 2: Generate mock reviews for each title
def generate_mock_reviews(titles):
    sample_reviews = [
        "Absolutely loved it! Highly recommended.",
        "It was okay, not the best but not the worst.",
        "Really disappointing. Wouldn’t recommend.",
        "Fantastic read. Would buy again!",
        "Mediocre and forgettable.",
        "Awful. Waste of time.",
        "Good book, nice pacing and plot.",
        "Neutral feelings. It was fine.",
        "Terrible writing and weak characters.",
        "Brilliant! Couldn't put it down!"
    ]
    return [(title, random.choice(sample_reviews)) for title in titles]

# Step 3: Sentiment Analysis
def analyze_sentiments(review_data):
    results = []
    for title, review in review_data:
        score = analyzer.polarity_scores(review)['compound']
        sentiment = 'Positive' if score > 0.05 else 'Negative' if score < -0.05 else 'Neutral'
        results.append({
            'Title': title,
            'Review': review,
            'Sentiment': sentiment,
            'Score': score
        })
    return pd.DataFrame(results)

# Run the full pipeline
titles = scrape_book_titles()
mock_reviews = generate_mock_reviews(titles)
df = analyze_sentiments(mock_reviews)

# Save to CSV
df.to_csv("book_sentiment_reviews.csv", index=False)
print("✅ Sentiment results saved to 'book_sentiment_reviews.csv'.")

# Visualize
plt.figure(figsize=(6, 4))
df['Sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
plt.title("Book Review Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Show samples
print("\n📚 Example Positive Review:")
print(df[df['Sentiment'] == 'Positive'].iloc[0], "\n")

print("📚 Example Negative Review:")
print(df[df['Sentiment'] == 'Negative'].iloc[0], "\n")

print("📚 Example Neutral Review:")
print(df[df['Sentiment'] == 'Neutral'].iloc[0], "\n")
