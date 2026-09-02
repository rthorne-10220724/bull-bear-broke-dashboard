import pandas as pd
import numpy as np
import ta

def calculate_indicators(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Calculates indicators identically for backtesting and live feeds."""
    if isinstance(df.columns, pd.MultiIndex) and symbol:
        df = df.xs(symbol, level=1, axis=1)
        
    df = df.copy()
    
    # 1. Technical Indicators
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['VOL_SMA20'] = df['Volume'].shift(1).rolling(20).mean()
    df['VOL_SPIKE'] = df['Volume'] > (1.2 * df['VOL_SMA20'])

    # 2. 4H Trend Simulation (Resampled from intraday data)
    df_4h = df.resample("4h", origin="start").agg({'Close': 'last'}).dropna()
    if len(df_4h) > 1:
        df_4h = df_4h.iloc[:-1]
        
    df_4h['EMA20'] = ta.trend.ema_indicator(df_4h['Close'], window=20)
    df_4h['EMA50'] = ta.trend.ema_indicator(df_4h['Close'], window=50)
    df_4h['TREND_4H'] = np.where(df_4h['EMA20'] > df_4h['EMA50'], "BULLISH", "BEARISH")
    
    df['TREND_4H'] = df_4h['TREND_4H'].reindex(df.index, method='ffill')
    return df

def check_signal(row: pd.Series) -> bool:
    """Returns True if exact entry conditions are met on the given bar."""
    return bool(
        row['RSI'] < 30 and 
        row['VOL_SPIKE'] and 
        row['TREND_4H'] == "BULLISH"
    )
