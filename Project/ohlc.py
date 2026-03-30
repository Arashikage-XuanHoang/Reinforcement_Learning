from binance.client import Client
import pandas as pd
import time
import os 

# Không cần API key vẫn dùng được historical data
client = Client()

# # symbol = "BTCUSDT"
# symbol = "XRPUSDT"
# interval = Client.KLINE_INTERVAL_1HOUR

# # Timestamp bắt đầu (Binance có từ ~2017)
# start_str = "1 Jan 2016"
# end_str = None  # None = tới hiện tại

# limit = 1000  # max của Binance

def crawl_cryto_data(symbol, limit, start_str, end_str, interval):
    all_data = []

    print(f"Start downloading {symbol} 1m data...")

    while True:
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_str,
            end_str=end_str,
            limit=limit
        )

        if not klines:
            break

        all_data.extend(klines)

        # cập nhật start_str = timestamp cuối + 1ms
        last_ts = klines[-1][0]
        start_str = last_ts + 1

        print(f"Downloaded {len(all_data)} rows...")

        # tránh rate limit
        time.sleep(0.3)

    print("Download complete!")

    # =========================
    # Convert sang DataFrame
    # =========================

    columns = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ]

    df = pd.DataFrame(all_data, columns=columns)

    # convert timestamp
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

    # chỉ giữ OHLCV
    df = df[["open_time","open","high","low","close","volume"]]

    # convert numeric
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)

    # =========================
    # Save file
    # =========================

    path = "/mnt/0498342198341420/UIT_DS_Third_Year_Second_Semester_2025_2026/Reinforcement_Learning/data/"
    file_name = f"{symbol}_1H_full.csv"
    full_path = os.path.join(path, file_name)

    df.to_csv(full_path, index=False)
    print(df.head())
    print(df.tail())
    print(f"Saved {symbol}")


interval = Client.KLINE_INTERVAL_1HOUR

# Timestamp bắt đầu (Binance có từ ~2017)
start_str = "1 Jan 2016"
end_str = None  # None = tới hiện tại

limit = 1000  # max của Binance
symbol_list = ["BNBUSDT", "BTCUSDT", "ETHUSDT", "LTCUSDT"] 

for i in range(len(symbol_list)):
    crawl_cryto_data(symbol_list[i], limit, start_str, end_str, interval)