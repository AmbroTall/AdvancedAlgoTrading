def calculate_stop_loss(entry_price, stop_loss_pips, instrument):
    if instrument == 'Forex':
        stop_loss_price = entry_price - (stop_loss_pips * 0.0001)  # Adjust pip value for Forex
        return stop_loss_price
    elif instrument == 'Stock':
        stop_loss_percentage = 0.01 * stop_loss_pips  # Assuming stop_loss_pips as percentage for stocks
        stop_loss_price = entry_price * (1 - stop_loss_percentage)
        return stop_loss_price
    else:
        raise ValueError("Instrument type not supported")

# Example usage:
entry_price = 1.2000  # Example entry price for Forex
stop_loss_pips = 10  # Example stop-loss in pips
stop_loss_price = calculate_stop_loss(entry_price, stop_loss_pips, 'Forex')
print(f'Stop Loss Price: {stop_loss_price}')
