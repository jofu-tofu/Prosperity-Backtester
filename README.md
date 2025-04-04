Few notes about using the backtester that I think are important.


# Simulating Bot Reactions

How it works: This backtester uses a second orderbook that simulates bot reactions to orders unfilled by the first orderbook given to us. This is done through taking logs that we get from prosperity and attaching them to the backtester to create an empirical distribution of what can happen at each timestamp. Get the most accurate results, I tried to attach logs that vary orders as much as possible, having as much mutual information with each other and the true distibution - whilest having large enough size to capture all the liquidity data. I usually attach logs for vwap_d{i}, which buys and sells at vwap+-i, while neutralizing position at each timestep. But I think the more logs you attach the better to be honest.

1. When setting bot behavior, I think there are only two that are useful given the structure of how I coded the liquidity orderbook. Setting it to "none" gets rid of the second orderbook. Setting it to "eq" makes it so that orders you submit can only be matched to the corresponding same price in the liquidity orderbook. I think this is the most accurate.