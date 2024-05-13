import yfinance as yf


def download_data(ticker, start_date, end_date, filename):
    """
    Download historical stock data from Yahoo Finance and save it to a CSV file.

    :param ticker: The ticker symbol of the stock (e.g., 'AAPL', 'GOOGL').
    :param start_date: The start date of the period (format: 'YYYY-MM-DD').
    :param end_date: The end date of the period (format: 'YYYY-MM-DD').
    :param filename: The path to the file where the data will be saved.
    """
    # Fetch data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)

    # Save the data to a CSV file
    data.to_csv(filename)
    print(f"Data for {ticker} saved to {filename}")


if __name__ == "__main__":
    # Example usage
    download_data('AGG', '2000-01-01', '2020-12-31', 'AGG.csv')
