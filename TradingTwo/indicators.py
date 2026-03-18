import pandas as pd 
import pandas_ta as ta 

def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """
    -> pd.DataFrame
    Đây là type hint cho giá trị trả về
    Hàm này được kỳ vọng trả về một pandas.DataFrame
    """

    # Load data từ đường dẫn và tiền xử lý bằng các technical indicators

    df = pd.read_csv(csv_path, parse_dates=True, index_col='Gmt time')
    # parse_dates=True nghĩa là gì?
    # 👉 Tự động chuyển các cột ngày/giờ từ dạng string sang kiểu datetime của pandas.

    # Sort by date 
    df.sort_index(inplace=True)

    # Các cột indicators làm ví dụ 
    df['rsi_14'] = ta.rsi(df['Close'], length=14)
    df['ma_20'] = ta.sma(df['Close'], length=20)
    df['ma_50'] = ta.sma(df['Close'], length=50)
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # Slope feature
    df['ma_20_slope']= df['ma_20'].diff()

    # Drop any rows with NaN
    df.dropna(inplace=True)

    return df