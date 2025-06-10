import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_books(num_pages=2):
    base_url = "http://books.toscrape.com/catalogue/page-{}.html"
    books = []

    for page in range(1, num_pages + 1):
        url = base_url.format(page)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        items = soup.select("article.product_pod")

        for book in items:
            title = book.h3.a['title']
            price = book.select_one("p.price_color").text.strip()
            rating = book.select_one("p.star-rating")['class'][1]  # e.g., 'Three'

            books.append({
                "Title": title,
                "Price": price,
                "Rating": rating
            })

    df = pd.DataFrame(books)
    df.to_csv("books_data.csv", index=False)
    print(f"✅ Saved {len(books)} books to books_data.csv")

scrape_books()
