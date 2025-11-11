"""
Data Audit and Quality Checks for Crypto Market Health Dashboard

This script performs automated data quality checks on the extracted and transformed
cryptocurrency data to ensure data integrity and reliability.

Audit Rules Implemented:
1. Missing Dates - Detect gaps in price or sentiment series
2. Duplicates - Ensure no duplicate dates × coin combinations
3. Negative Values - Flag invalid price/volume < 0
4. Outliers - Identify > 3 σ volatility anomalies
5. Data Consistency - Verify all coins have similar date ranges
6. Data Freshness - Check if data is up to date
7. Value Range Validation - Validate data falls within expected ranges

Author: Crypto Analytics Team
Date: 2025
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import logging

# Set up logging for monitoring and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database file path
db_path = "crypto_analytics.db"
# Connect to SQLite database
conn = sqlite3.connect(db_path)

# Initialize list to store audit results
rules = []
# Get current timestamp for audit log
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

try:
    # --- AUDIT RULE 1: Missing Dates ---
    # Detect gaps in the date series which could indicate data collection issues
    logger.info("Checking for missing dates...")
    df = pd.read_sql("SELECT DISTINCT date FROM combined_metrics", conn)
    if not df.empty:
        # Generate expected date range
        expected = pd.date_range(df["date"].min(), df["date"].max())
        # Calculate missing dates
        missing = len(set(expected) - set(pd.to_datetime(df["date"])))
        rules.append(("Missing Dates", len(expected), missing))
        logger.info(f"Missing dates check: Expected {len(expected)}, Missing {missing}")
    else:
        logger.warning("No data found for missing dates check")
        rules.append(("Missing Dates", 0, 0))

    # --- AUDIT RULE 2: Duplicate Records ---
    # Ensure no duplicate date-symbol combinations which would indicate data processing errors
    logger.info("Checking for duplicate records...")
    dupes = pd.read_sql("""
    SELECT date, symbol, COUNT(*) as cnt 
    FROM combined_metrics 
    GROUP BY date, symbol HAVING cnt > 1
    """, conn)
    rules.append(("Duplicate Records", len(dupes), len(dupes)))
    logger.info(f"Duplicate records check: Found {len(dupes)} duplicates")

    # --- AUDIT RULE 3: Negative Values ---
    # Flag invalid negative prices or volumes which would indicate data quality issues
    logger.info("Checking for negative values...")
    negatives = pd.read_sql("SELECT * FROM price_data WHERE Close < 0 OR Volume < 0", conn)
    rules.append(("Negative Values", len(negatives), len(negatives)))
    logger.info(f"Negative values check: Found {len(negatives)} records")

    # --- AUDIT RULE 4: Volatility Outliers ---
    # Identify volatility anomalies (> 3σ) which could indicate extreme market events
    logger.info("Checking for volatility outliers...")
    outliers = pd.read_sql("""
    SELECT * FROM combined_metrics 
    WHERE volatility > (
        SELECT AVG(volatility) + 3 * (
            SELECT AVG(volatility * volatility) - AVG(volatility) * AVG(volatility)
            FROM combined_metrics
        ) FROM combined_metrics
    )
    """, conn)
    rules.append(("Volatility Outliers", len(outliers), len(outliers)))
    logger.info(f"Volatility outliers check: Found {len(outliers)} outliers")

    # --- AUDIT RULE 5: Data Consistency ---
    # Verify all cryptocurrencies have similar date ranges for consistent analysis
    logger.info("Checking data consistency...")
    coin_counts = pd.read_sql("""
    SELECT symbol, COUNT(*) as count 
    FROM combined_metrics 
    GROUP BY symbol
    """, conn)
    
    if not coin_counts.empty:
        min_count = coin_counts["count"].min()
        max_count = coin_counts["count"].max()
        consistency_issues = max_count - min_count
        rules.append(("Data Consistency", len(coin_counts), consistency_issues))
        logger.info(f"Data consistency check: Min count {min_count}, Max count {max_count}")
    else:
        rules.append(("Data Consistency", 0, 0))
        logger.warning("No data found for consistency check")

    # --- AUDIT RULE 6: Data Freshness ---
    # Check if the data is up to date (should be updated daily)
    logger.info("Checking data freshness...")
    latest_date_result = conn.execute("SELECT MAX(date) FROM combined_metrics").fetchone()
    if latest_date_result[0]:
        latest_date = datetime.strptime(latest_date_result[0], '%Y-%m-%d %H:%M:%S')
        days_since_update = (datetime.now() - latest_date).days
        rules.append(("Data Freshness", days_since_update, 1 if days_since_update > 2 else 0))
        logger.info(f"Data freshness check: Latest data is {days_since_update} days old")
    else:
        rules.append(("Data Freshness", 0, 1))
        logger.warning("No data found for freshness check")

    # --- AUDIT RULE 7: Value Range Validation ---
    # Validate that data falls within expected ranges
    logger.info("Checking value ranges...")
    range_issues = 0
    
    # Check fear_greed values are within 0-100 range
    invalid_sentiment = pd.read_sql("""
    SELECT * FROM sentiment_index 
    WHERE fear_greed < 0 OR fear_greed > 100
    """, conn)
    range_issues += len(invalid_sentiment)
    
    # Check volatility is not negative
    negative_volatility = pd.read_sql("""
    SELECT * FROM combined_metrics 
    WHERE volatility < 0
    """, conn)
    range_issues += len(negative_volatility)
    
    rules.append(("Value Range Validation", 2, range_issues))
    logger.info(f"Value range validation: Found {range_issues} range issues")

    # --- LOG RESULTS ---
    # Store audit results in database for dashboard display
    log_df = pd.DataFrame(rules, columns=["rule_name", "record_count", "issues_found"])
    log_df["timestamp"] = now
    log_df.to_sql("audit_log", conn, if_exists="append", index=False)
    logger.info("Audit results:")
    print(log_df)
    
    # Provide summary
    total_issues = log_df["issues_found"].sum()
    if total_issues > 0:
        logger.warning(f"Audit completed with {total_issues} issues found.")
    else:
        logger.info("Audit completed successfully with no issues found.")

except Exception as e:
    # Log any errors that occur during audit checks
    logger.error(f"Error during audit checks: {e}")

finally:
    # Always close database connection
    conn.close()
    logger.info("Audit completed.")