"""
ETL Pipeline for Crypto Market Health & Sentiment Dashboard

This script extracts cryptocurrency price data and market sentiment data,
transforms it into meaningful metrics, and loads it into a SQLite database.

Business Question Addressed:
"How do price volatility, trading volume, and social sentiment correlate 
with investor confidence across major cryptocurrencies?"

Data Sources:
1. Yahoo Finance (yfinance) - Historical OHLCV data for cryptocurrencies
2. Alternative.me Crypto Fear & Greed Index API - Market sentiment data

Author: Crypto Analytics Team
Date: 2025
"""

import pandas as pd
import yfinance as yf
import requests
import sqlite3
from datetime import datetime, timedelta
import logging
import time
from typing import Optional
from tqdm import tqdm
import concurrent.futures

# Set up logging for monitoring and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# List of cryptocurrencies to analyze
cryptos = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "DOGE-USD"]
# Start date for historical data (2 years ago)
start_date = "2020-01-01"
# Database file path
db_path = "crypto_analytics.db"

def get_last_update_date(conn: sqlite3.Connection) -> Optional[datetime]:
    """
    Get the last update date from the database to enable incremental updates.
    
    Args:
        conn (sqlite3.Connection): Database connection
        
    Returns:
        datetime: Last update date, or None if no data exists
    """
    try:
        result = conn.execute("SELECT MAX(date) FROM combined_metrics").fetchone()
        if result[0]:
            return datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S')
        return None
    except Exception as e:
        logger.warning(f"Could not retrieve last update date: {e}")
        return None

def fetch_yahoo_finance_data(symbol: str, start_date: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
    """
    Fetch cryptocurrency data from Yahoo Finance with retry mechanism.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., "BTC-USD")
        start_date (str): Start date for historical data in YYYY-MM-DD format
        max_retries (int): Maximum number of retry attempts
    
    Returns:
        pd.DataFrame: DataFrame containing cryptocurrency price data, or None if failed
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching {symbol} data (attempt {attempt + 1}/{max_retries})")
            df = yf.download(symbol, start=start_date)
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            return df
        except Exception as e:
            logger.error(f"Error fetching {symbol} data (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to fetch {symbol} data after {max_retries} attempts")
    return None

def fetch_single_crypto_data(symbol: str, start_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch data for a single cryptocurrency and prepare it for concatenation.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., "BTC-USD")
        start_date (str): Start date for historical data in YYYY-MM-DD format
    
    Returns:
        pd.DataFrame: Formatted DataFrame with cryptocurrency data, or None if failed
    """
    df = fetch_yahoo_finance_data(symbol, start_date)
    if df is not None:
        # Flatten multi-index columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)  # Remove the symbol level from columns
        df["symbol"] = symbol
        return df.reset_index()
    return None

def fetch_fear_greed_index(max_retries: int = 3) -> Optional[pd.DataFrame]:
    """
    Fetch Fear & Greed Index from Alternative.me API with retry mechanism.
    
    Args:
        max_retries (int): Maximum number of retry attempts
    
    Returns:
        pd.DataFrame: DataFrame containing sentiment data, or None if failed
    """
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching Fear & Greed Index (attempt {attempt + 1}/{max_retries})")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            fng_data = response.json()["data"]
            sentiment_df = pd.DataFrame(fng_data)
            sentiment_df["date"] = pd.to_datetime(sentiment_df["timestamp"], unit='s')
            sentiment_df = sentiment_df[["date", "value"]].rename(columns={"value": "fear_greed"})
            sentiment_df["fear_greed"] = sentiment_df["fear_greed"].astype(int)
            return sentiment_df
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to fetch Fear & Greed Index after {max_retries} attempts")
    return None

# --- EXTRACT PHASE ---
logger.info("Starting ETL pipeline...")

# For now, we'll do a full refresh to avoid the complexity of incremental updates
# In a production environment, you might want to implement proper incremental updates
logger.info(f"Performing full data fetch from {start_date}")

# Extract price data for all cryptocurrencies using parallel processing
logger.info("Extracting price data...")
price_data = []

# Use ThreadPoolExecutor for parallel data fetching to improve performance
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # Submit all tasks
    future_to_symbol = {executor.submit(fetch_single_crypto_data, symbol, start_date): symbol for symbol in cryptos}
    
    # Collect results with progress bar
    for future in tqdm(concurrent.futures.as_completed(future_to_symbol), total=len(cryptos), desc="Fetching Crypto Data"):
        symbol = future_to_symbol[future]
        try:
            df = future.result()
            if df is not None:
                price_data.append(df)
            else:
                logger.warning(f"Failed to fetch data for {symbol}")
        except Exception as e:
            logger.error(f"Exception occurred for {symbol}: {e}")

# Exit if no price data was fetched
if not price_data:
    logger.error("No price data fetched for any cryptocurrency. Exiting.")
    exit(1)

# Combine all cryptocurrency data into a single DataFrame
price_df = pd.concat(price_data)
price_df.rename(columns={"Date": "date"}, inplace=True)

# Extract Fear & Greed Index sentiment data
logger.info("Fetching sentiment data...")
sentiment_df = fetch_fear_greed_index()
if sentiment_df is None:
    logger.error("Failed to fetch sentiment data. Exiting.")
    exit(1)

# --- TRANSFORM PHASE ---
logger.info("Transforming data...")
# Data validation to ensure we have data to work with
if price_df.empty:
    logger.error("Price data is empty. Exiting.")
    exit(1)

if sentiment_df.empty:
    logger.error("Sentiment data is empty. Exiting.")
    exit(1)

# Remove any duplicate dates for the same symbol to prevent concatenation issues
price_df = price_df.drop_duplicates(subset=['date', 'symbol'], keep='first')

# Calculate financial metrics for each cryptocurrency using the working approach
returns_data = []
# Process each cryptocurrency with progress tracking
for symbol in tqdm(cryptos, desc="Processing Cryptocurrencies"):
    symbol_data = price_df[price_df["symbol"] == symbol].copy()
    if not symbol_data.empty:
        # Calculate daily returns (percentage change in closing price) using transform
        symbol_data["returns"] = symbol_data["Close"].pct_change()
        # Calculate volatility (7-day rolling standard deviation of returns)
        symbol_data["volatility"] = symbol_data["returns"].rolling(7).std()
        # Calculate volume change (percentage change in trading volume)
        symbol_data["volume_change"] = symbol_data["Volume"].pct_change()
        returns_data.append(symbol_data)
    else:
        logger.warning(f"No data found for {symbol}")

# Exit if no valid data after transformation
if not returns_data:
    logger.error("No valid data after transformation. Exiting.")
    exit(1)

price_df = pd.concat(returns_data)

# Merge price data with sentiment data by date
merged_df = pd.merge(price_df, sentiment_df, on="date", how="left")

# Calculate derived market health score
# Market Health = Volume × (1 - Volatility) × (FearGreed/100)
merged_df["market_health"] = (
    merged_df["Volume"].fillna(0) * 
    (1 - merged_df["volatility"].fillna(0)) * 
    (merged_df["fear_greed"].fillna(50) / 100)
)

# --- DATA QUALITY CHECKS ---
logger.info("Performing data quality checks...")
# Check for missing values in each dataset
missing_price_data = price_df.isnull().sum()
missing_sentiment_data = sentiment_df.isnull().sum()
missing_combined_data = merged_df.isnull().sum()

logger.info(f"Missing values in price data:\n{missing_price_data}")
logger.info(f"Missing values in sentiment data:\n{missing_sentiment_data}")
logger.info(f"Missing values in combined data:\n{missing_combined_data}")

# Check for negative prices or volumes which would indicate data issues
negative_prices = price_df[price_df["Close"] < 0]
negative_volumes = price_df[price_df["Volume"] < 0]

if not negative_prices.empty:
    logger.warning(f"Found {len(negative_prices)} records with negative prices")
    
if not negative_volumes.empty:
    logger.warning(f"Found {len(negative_volumes)} records with negative volumes")

# --- LOAD PHASE ---
logger.info("Loading data into database...")
# Connect to SQLite database
conn = sqlite3.connect(db_path)

try:
    # Store processed data in separate tables
    price_df.to_sql("price_data", conn, if_exists="replace", index=False)
    sentiment_df.to_sql("sentiment_index", conn, if_exists="replace", index=False)
    merged_df.to_sql("combined_metrics", conn, if_exists="replace", index=False)
    logger.info("Data successfully loaded into database")
except Exception as e:
    logger.error(f"Error loading data into database: {e}")
    conn.close()
    exit(1)

# Close database connection
conn.close()
logger.info("ETL complete. Data stored in crypto_analytics.db")