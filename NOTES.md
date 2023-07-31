## Ways of doing train test
- Cross validation
- Random Seed


## Advantages of RandomForestClassifier
- Trains a bunch of individual decision trees with randomized parameters and avarages the results from those decision trees, hence, resistant to overfitting
- Run relatively quickly
- Can pick non-linear relationships in the data

## n_estimators
This is the number of individual decision tress to be trained. The higher the number the better accuracy

## min_samples_split, 100
Helps to protect against overfitting

## Summary
Other than looking at time series data only, there are other factors to determine stock prices, such as,
- Technology crashes
- Corporate News
- Macro-economic conditions like: interest rates, inflatuations, 
- Hourly or Minute per minute data
