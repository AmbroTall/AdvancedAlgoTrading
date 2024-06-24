import time
from datetime import datetime
from ib_insync import *
import pandas as pd
import numpy as np

# Connect to IB TWS
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define the initial balance for demo purposes
initial_balance = 100000

# Function to get the current account balance
def get_account_balance():
    # For simulation purposes, return the initial balance
    return initial_balance

# Function to reset the account balance
def reset_account_balance():
    global initial_balance
    initial_balance = 10000
    print(f"Account balance reset to {initial_balance}")

# Main function for your trading logic
def main():
    global initial_balance
    current_balance = get_account_balance()
    print(f"Current balance: {current_balance}")

    # Implement your trading logic here
    # ...

# Reset the balance
reset_account_balance()

# Run the main function
main()

# Disconnect from IB
ib.disconnect()
