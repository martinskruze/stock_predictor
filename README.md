# stock predictor
## task
Lets predict stock movement after 5 trading days using 30 day stock prices!

##  input data
You are given 10 years of WDAY stock data.
Each record consists of following data points
1. 30 days of data of pricepoints (6 per day) [pre_market, open, high, low, close, after_hours]
2. label of price change taking in 5 days close price after last day's close with these labels
    0. "0: Strong rise (more than $10 increase)",
    1. "1: Moderate rise ($5 to $10 increase)",
    2. "2: Slight rise ($1 to $5 increase)",
    3. "3: Stable (within $1 change)",
    4. "4: Slight drop ($1 to $5 decrease)",
    5. "5: Moderate drop ($5 to $10 decrease)",
    6. "6: Strong drop (more than $10 decrease)"

## starting and running the application
### install dependencies:
```
pip install -r requirements.txt
```

### train models
```
python3 train.py --model_name linear 
python3 train.py --model_name mlp  
python3 train.py --model_name mlp_deep 
python3 train.py --model_name mlp_deep_residual
```

### check on progress
```
tensorboard --logdir logs
```