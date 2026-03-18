import pandas as pd 
import pandas_ta as ta

def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """
    Loads EUR/USD data from CSV and preprocesses it by adding technical indicators.
    Expects columns: [Gmt time, Open, High, Low, Close, Volume]

    """

    df = pd.read_csv(csv_path, parse_dates=True, index_col='Gmt time')

    # Sort by date just in case 
    df.sort_index(inplace=True)

    # Example technical indicators from pandas_ta
    df['rsi_14'] = ta.rsi(df['Close'], length=14)
    df['ma_20'] = ta.sma(df['Close'], length=20)
    df['ma_50'] = ta.sma(df['Close'], length=50)
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # Slope feature
    df['ma_20_slope']= df['ma_20'].diff()

    # Drop any rows with NaN
    df.dropna(inplace=True)

    return df