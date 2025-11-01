import asyncio
import csv
from datetime import datetime
from random import randint

from twikit import Client, TooManyRequests

MINIMUM_TWEETS = 5000
YEAR = 2024
QUERY = f"(from:ElonMusk) tesla lang:en until:{YEAR + 1}-01-01 since:{YEAR}-01-01"


async def scrape_tweets():
    client = Client(language="en-US")
    client.load_cookies("cookies.json")

    # Create CSV file with headers
    filename = f"elonmusk_tesla_tweets_{YEAR}.csv"
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["ID", "Username", "Text", "Date", "Retweets", "Likes"])

    tweet_count = 0
    tweets = None

    while tweet_count < MINIMUM_TWEETS:
        try:
            if tweets is None:
                print("Starting new search...")
                tweets = await client.search_tweet(QUERY, product="Top")
            else:
                wait_time = randint(5, 15)
                print(f"Waiting for {wait_time} seconds before fetching more tweets...")
                await asyncio.sleep(wait_time)
                tweets = await tweets.next()

        except TooManyRequests as e:
            if e.rate_limit_reset is not None:
                rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
                print(f"Rate limit exceeded. Try again at {rate_limit_reset}.")
                sleep_duration = (rate_limit_reset - datetime.now()).total_seconds()
                await asyncio.sleep(sleep_duration + 1)
                continue

        if not tweets:
            print("No more tweets found. Exiting.")
            break

        for tweet in tweets:
            tweet_data = [
                tweet.id,
                tweet.user.name,
                tweet.text.replace("\n", " "),  # Remove newlines for CSV
                tweet.created_at,
                tweet.retweet_count,
                tweet.favorite_count,
            ]

            with open(filename, "a", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(tweet_data)

            tweet_count += 1
            if tweet_count >= MINIMUM_TWEETS:
                break

        print(f"Saved {tweet_count} tweets so far...")

    print(f"Finished! Total tweets saved: {tweet_count}")


if __name__ == "__main__":
    asyncio.run(scrape_tweets())
