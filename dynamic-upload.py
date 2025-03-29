import json
import os
import datetime
import finnhub
from google.cloud import storage
from time import sleep
from secret import FINHUB_KEY
from google.oauth2 import service_account
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import requests

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=FINHUB_KEY)

# Initialize Google Cloud Storage client
# credentials_path = "D:/Coding/IntroML/coe379-ml-project-81b7de97df4a.json"
credentials_path = "C:/Users/arish/Coding/MLClass/coe379-ml-project-55cd4cde117a.json"

credentials = service_account.Credentials.from_service_account_file(credentials_path)
storage_client = storage.Client(credentials=credentials)

# Define bucket and directories for each company
BUCKET_NAME = "fin_analysis_data"
COMPANIES = {
    "RTX": "raytheon/news_articles/to_delete",
    "NOC": "northrop/news_articles/to_delete",
    "LMT": "lockheed/news_articles/to_delete",
}


def get_finnhub_news(company, start_date, end_date):
    """Fetch news articles for a specific company from Finnhub"""
    news_data = finnhub_client.company_news(company, _from=start_date, to=end_date)
    filtered_news = [
        {
            "headline": item.get("headline", ""),
            "summary": item.get("summary", ""),
            "source": item.get("source", ""),
            "url": item.get("url", ""),
        }
        for item in news_data
    ]

    return filtered_news

def get_alpha_news(company, start_date, end_date):
    """Fetch news articles for a specific company from Alpha Vantage"""
    # TODO: Implement fetching news articles from Alpha Vantage
    url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey=IXAHBNDC1EB314QW'
    r = requests.get(url)
    data = r.json()

    print(data)
    pass


def save_to_gcs(company, data, file_date):
    """Save the news data as a JSON file to Google Cloud Storage"""
    # Create the JSON file name based on the specific date
    file_name = f"{company}_{file_date}_news.json"

    # Convert data to JSON string
    json_data = json.dumps(data, indent=4)

    # Define the blob (file) path
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"fin_text/{COMPANIES[company]}/{file_name}")

    # Upload JSON data to the bucket
    blob.upload_from_string(json_data, content_type="application/json")
    print(f"Uploaded {file_name} to gs://{BUCKET_NAME}/{COMPANIES[company]}/")

def save_to_mongo(company, data, file_date):
    load_dotenv()
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client["stocks"]
    collection = db["news"]
    entry = {
                    "ticker": company,
                    "date": file_date,
                    "news": data
                }
    insert_result = collection.insert_one(entry)
    print("Inserted Document ID: ", insert_result)


# def main():
#     # Loop over each company to fetch and upload news
#     for company in COMPANIES.keys():
#         print(f"Processing news for {company}...")

#         # Loop over the last 20 days
#         for day_offset in range(20):
#             try:
#                 # Define the date range for the current day
#                 end_date = datetime.datetime.now() - datetime.timedelta(days=day_offset)
#                 start_date = end_date - datetime.timedelta(days=1)

#                 # Format the dates as strings (YYYY-MM-DD)
#                 start_date_str = start_date.strftime("%Y-%m-%d")
#                 end_date_str = end_date.strftime("%Y-%m-%d")

#                 print(
#                     f"Fetching news for {company} from {start_date_str} to {end_date_str}..."
#                 )
#                 news_data = get_finnhub_news(company, start_date_str, end_date_str)

#                 if news_data:
#                     print(f"Saving {len(news_data)} articles for {company} to GCS...")
#                     save_to_gcs(company, news_data, start_date_str)
#                 else:
#                     print(f"No news articles found for {company} on {start_date_str}.")

#             except Exception as e:
#                 print(f"Error processing {company} for {start_date_str}: {e}")

#             # Sleep to avoid hitting API rate limits (if necessary)
#             sleep(5)


# if __name__ == "__main__":
#     main()


def main():
    # Define the specific start and end dates
    start_overall = datetime.datetime(2024, 10, 1)
    end_overall = datetime.datetime(2024, 11, 22)

    # Loop over each company to fetch and upload news
    for company in COMPANIES.keys():
        print(f"Processing news for {company}...")

        # Calculate total days to iterate
        total_days = (end_overall - start_overall).days + 1

        # Loop over the entire date range
        for day_offset in range(total_days):
            try:
                # Calculate current date
                current_date = start_overall + datetime.timedelta(days=day_offset)

                # Define the date range for the current day
                start_date = current_date
                end_date = current_date + datetime.timedelta(days=1)

                # Format the dates as strings (YYYY-MM-DD)
                start_date_str = start_date.strftime("%Y-%m-%d")
                end_date_str = end_date.strftime("%Y-%m-%d")

                print(
                    f"Fetching news for {company} from {start_date_str} to {end_date_str}..."
                )
                news_data = get_finnhub_news(company, start_date_str, end_date_str)

                if news_data:
                    print(f"Saving {len(news_data)} articles for {company} to GCS...")
                    # saving to gcloud
                    # save_to_gcs(company, news_data, start_date_str)
                    # saving to mongo
                    save_to_mongo(company, news_data, start_date_str)
                else:
                    print(f"No news articles found for {company} on {start_date_str}.")

            except Exception as e:
                print(f"Error processing {company} for {start_date_str}: {e}")

            # Sleep to avoid hitting API rate limits
            sleep(5)


if __name__ == "__main__":
    main()
