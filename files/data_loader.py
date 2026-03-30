import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Optional
import time

class BinanceDataLoader:
    """
    Load OHLCV data từ Binance API
    """
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3/klines"
        
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Lấy dữ liệu OHLCV từ Binance
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            start_time: Thời gian bắt đầu
            end_time: Thời gian kết thúc
            limit: Số lượng candles (max 1000)
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Select relevant columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_historical_data(
        self,
        symbol: str,
        interval: str = "1m",
        days: int = 7
    ) -> pd.DataFrame:
        """
        Lấy dữ liệu lịch sử cho nhiều ngày
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            days: Số ngày lấy dữ liệu
        """
        all_data = []
        end_time = datetime.now()
        
        # Binance giới hạn 1000 candles mỗi request
        # Với 1m interval: 1000 candles = ~16.67 hours
        candles_per_request = 1000
        
        if interval == "1m":
            hours_per_request = candles_per_request / 60
        elif interval == "5m":
            hours_per_request = candles_per_request * 5 / 60
        elif interval == "15m":
            hours_per_request = candles_per_request * 15 / 60
        elif interval == "1h":
            hours_per_request = candles_per_request
        else:
            hours_per_request = candles_per_request / 60
        
        total_requests = int((days * 24) / hours_per_request) + 1
        
        print(f"Fetching {days} days of {interval} data for {symbol}...")
        print(f"Total requests needed: {total_requests}")
        
        for i in range(total_requests):
            request_end_time = end_time - timedelta(hours=i * hours_per_request)
            request_start_time = request_end_time - timedelta(hours=hours_per_request)
            
            df = self.fetch_ohlcv(
                symbol=symbol,
                interval=interval,
                start_time=request_start_time,
                end_time=request_end_time,
                limit=candles_per_request
            )
            
            if not df.empty:
                all_data.append(df)
            
            # Rate limiting
            time.sleep(0.2)
            
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{total_requests} requests completed")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['timestamp'])
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            print(f"Total candles fetched: {len(combined_df)}")
            return combined_df
        else:
            return pd.DataFrame()
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        interval: str = "1m",
        days: int = 7
    ) -> dict:
        """
        Lấy dữ liệu cho nhiều symbols
        
        Returns:
            Dictionary với key là symbol và value là DataFrame
        """
        data_dict = {}
        
        for symbol in symbols:
            print(f"\n{'='*60}")
            print(f"Fetching data for {symbol}")
            print(f"{'='*60}")
            
            df = self.fetch_historical_data(
                symbol=symbol,
                interval=interval,
                days=days
            )
            
            if not df.empty:
                data_dict[symbol] = df
                print(f"✓ Successfully fetched {len(df)} candles for {symbol}")
            else:
                print(f"✗ Failed to fetch data for {symbol}")
            
            time.sleep(1)  # Rate limiting giữa các symbols
        
        return data_dict
    
    def save_to_csv(self, data_dict: dict, output_dir: str = "."):
        """
        Lưu dữ liệu vào CSV files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for symbol, df in data_dict.items():
            filename = f"{output_dir}/{symbol}_{df['timestamp'].min().strftime('%Y%m%d')}_{df['timestamp'].max().strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved {symbol} data to {filename}")
    
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load dữ liệu từ CSV
        """
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df


def prepare_training_data(
    symbols: List[str],
    interval: str = "1m",
    days: int = 7,
    save_csv: bool = True
) -> dict:
    """
    Hàm tiện ích để chuẩn bị dữ liệu training
    """
    loader = BinanceDataLoader()
    
    print(f"\n{'='*70}")
    print(f"PREPARING TRAINING DATA")
    print(f"{'='*70}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Interval: {interval}")
    print(f"Days: {days}")
    print(f"{'='*70}\n")
    
    data_dict = loader.fetch_multiple_symbols(
        symbols=symbols,
        interval=interval,
        days=days
    )
    
    if save_csv:
        loader.save_to_csv(data_dict, output_dir="data")
    
    print(f"\n{'='*70}")
    print(f"DATA PREPARATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total symbols fetched: {len(data_dict)}")
    for symbol, df in data_dict.items():
        print(f"  {symbol}: {len(df)} candles")
    print(f"{'='*70}\n")
    
    return data_dict


if __name__ == "__main__":
    # Example usage
    symbol_list = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "LTCUSDT", "XRPUSDT"]
    
    data = prepare_training_data(
        symbols=symbol_list,
        interval="1m",
        days=7,
        save_csv=True
    )
