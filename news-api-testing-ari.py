import json
import requests


def get_alpha_news(company, start_date, end_date):
    """Fetch news articles for a specific company from Alpha Vantage"""
    # TODO: Implement fetching news articles from Alpha Vantage
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={company}&topics=technology&apikey=IXAHBNDC1EB314QW"
    r = requests.get(url)
    data = r.json()

    print(data)
    return data

with open("test.json", "w") as f:
    json.dump(get_alpha_news("AAPL", ".", "."), f, indent=4)

# print(get_alpha_news("AAPL", ".", "."))
