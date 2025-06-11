import pandas as pd
import pyodbc
import nltk
import warnings
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import os
load_dotenv()
nltk.download('vader_lexicon')

def fetch_data_from_sql():
    # Fetch values from environment variables
    driver = os.getenv("DB_DRIVER")
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_NAME")
    trusted_connection = os.getenv("DB_TRUSTED_CONNECTION", "yes")

    # Build the connection string
    conn_str = (
        f"Driver={driver};"
        f"Server={server};"
        f"Database={database};"
        f"Trusted_Connection={trusted_connection};"
    )
    try:
        conn = pyodbc.connect(conn_str)
        print("Connected to SQL Server successfully.")
        query = "SELECT ReviewID, CustomerID, ProductID, ReviewDate, Rating, ReviewText FROM dbo.customer_reviews"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
        
        #Code to Filter Warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="pandas.io.sql")
            
    except pyodbc.Error as e:
        print("Error connecting to SQL Server:", e)

customer_reviews_df = fetch_data_from_sql()

# Initialize the Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()   

def calculate_sentiment(review):
    #Polarity Scores gives a dictionary with negative, neutral, positive, and compound scores,
    # we want the compound score
    sentiment = sia.polarity_scores(review)
    return sentiment['compound']

def categorize_sentiment(score, rating):
    if score >= 0.05:
        if rating >= 4:
            return 'Highly Positive'
        elif rating == 3:
            return 'Moderately Positive'
        else:
            return 'Mixed Neutral'
        
    elif score <= -0.05:      
        if rating <= 2:
            return 'Highly Negative'
        elif rating == 3:
            return 'Moderately Negative'
        else:
            return 'Mixed Neutral'
        
    else:
        return 'Neutral'
    
def sentiment_bucket(score):
    if score >= 0.5:
        return '0.5 to 1.0'  #Indicates strongly positive sentiment
    elif 0.0 <= score <0.5:
        return '0.0 to 0.49' #Indicates moderately positive sentiment
    elif -0.5 <= score < 0.0:
        return '-0.49 to 0.0'  #Indicates moderately negative sentiment
    else:
        return '-1.0 to -0.5' #Indicates strongly negative sentiment
    

#Create a dataframe 'SentimentScore' and apply the sentiment analysis function
customer_reviews_df['SentimentScore'] = customer_reviews_df['ReviewText'].apply(calculate_sentiment)

#Apply sentiment categorization
customer_reviews_df['SentimentCategory'] = customer_reviews_df.apply(
    lambda row: categorize_sentiment(row['SentimentScore'], row['Rating']), axis=1
)

customer_reviews_df['SentimentBucket'] = customer_reviews_df['SentimentScore'].apply(sentiment_bucket)

print(customer_reviews_df.head())

#Save the processed DataFrame to a CSV file
customer_reviews_df.to_csv('factored_customer_reviews.csv', index=False)