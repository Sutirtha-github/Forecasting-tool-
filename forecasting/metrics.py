def calculate_mape(actual, predicted):
    merged = actual.rename(columns={'y' : 'actual'}).merge(predicted.rename(columns={'y':'predicted'}), on='ds')
    merged['mape'] = 100*abs(merged['actual'] - merged['predicted']) / merged['actual'].clip(lower=1)
    return merged['mape'].mean(), merged