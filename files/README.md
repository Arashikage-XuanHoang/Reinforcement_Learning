# Crypto Trading Bot with Reinforcement Learning

Hệ thống Reinforcement Learning để train trading bot trên các cặp cryptocurrency với dữ liệu OHLCV interval 1m.

## 🚀 Tính năng

- **Multi-symbol training**: Train trên nhiều cặp crypto cùng lúc (BTCUSDT, ETHUSDT, BNBUSDT, LTCUSDT, XRPUSDT)
- **PPO Algorithm**: Sử dụng Proximal Policy Optimization từ Stable-Baselines3
- **Technical Indicators**: Tích hợp RSI, MACD, Bollinger Bands, Moving Averages
- **Flexible Actions**: 7 actions (Hold, Buy 25%/50%/100%, Sell 25%/50%/100%)
- **Real-time Data**: Tự động fetch dữ liệu từ Binance API
- **Backtesting**: Test và đánh giá model với visualization chi tiết
- **Tensorboard Integration**: Monitor training progress real-time

## 📋 Yêu cầu

```bash
Python 3.8+
pip install -r requirements.txt
```

## 🔧 Cài đặt

```bash
# Clone hoặc tải về các file
git clone <repo_url>
cd crypto-rl-trading

# Cài đặt dependencies
pip install -r requirements.txt
```

## 📊 Cấu trúc project

```
crypto-rl-trading/
├── trading_env.py          # Gym environment cho trading
├── data_loader.py          # Load dữ liệu từ Binance
├── train_rl.py            # Main training script
├── test_bot.py            # Testing và backtesting
├── requirements.txt       # Dependencies
├── data/                  # Cached data (tự động tạo)
├── models/                # Trained models (tự động tạo)
│   └── checkpoints/       # Training checkpoints
└── logs/                  # Training logs (tự động tạo)
    ├── tensorboard/       # Tensorboard logs
    └── training_history.json
```

## 🎯 Cách sử dụng

### 1. Training Model

```python
python train_rl.py
```

Script này sẽ:
1. Tự động fetch dữ liệu từ Binance (hoặc load từ cache)
2. Chuẩn bị training và evaluation environments
3. Train PPO model
4. Lưu model và checkpoints
5. Evaluate model performance

**Cấu hình training** (trong `train_rl.py`):

```python
symbol_list = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "LTCUSDT", "XRPUSDT"]
interval = "1m"
days = 7                    # Số ngày dữ liệu
total_timesteps = 200000    # Số timesteps training
learning_rate = 3e-4
```

### 2. Monitor Training (Real-time)

Mở terminal mới và chạy:

```bash
tensorboard --logdir logs/tensorboard
```

Truy cập http://localhost:6006 để xem:
- Training rewards
- Profit percentage
- Number of trades
- Loss curves

### 3. Testing/Backtesting

```python
python test_bot.py
```

Hoặc trong code:

```python
from test_bot import test_on_symbol, test_multiple_symbols

# Test trên 1 symbol
results = test_on_symbol(
    model_path="models/ppo_crypto_final_20250326_120000.zip",
    symbol="BTCUSDT",
    vecnormalize_path="models/ppo_crypto_final_20250326_120000_vecnormalize.pkl"
)

# Test trên nhiều symbols
all_results = test_multiple_symbols(
    model_path="models/ppo_crypto_final_20250326_120000.zip",
    symbol_list=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    vecnormalize_path="models/ppo_crypto_final_20250326_120000_vecnormalize.pkl"
)
```

### 4. Sử dụng Model đã train

```python
from test_bot import TradingBot
from data_loader import BinanceDataLoader

# Load model
bot = TradingBot(
    model_path="models/ppo_crypto_final_20250326_120000.zip",
    vecnormalize_path="models/ppo_crypto_final_20250326_120000_vecnormalize.pkl"
)

# Load data
loader = BinanceDataLoader()
df = loader.fetch_historical_data(symbol="BTCUSDT", interval="1m", days=3)

# Backtest
results = bot.backtest(df=df, initial_balance=10000.0)

# Visualize results
bot.plot_backtest_results(results, save_path="backtest_results.png")
```

## 📈 Environment Details

### Observation Space

Mỗi observation là một window của 50 timesteps, mỗi timestep có 15 features:

1. **OHLCV normalized**: Open, High, Low, Close, Volume
2. **Technical Indicators**:
   - RSI (Relative Strength Index)
   - MACD và MACD Signal
   - Bollinger Bands (Upper, Middle, Lower)
   - SMA 20 và EMA 20
   - Volume SMA
3. **Portfolio Info**: Current position ratio

### Action Space

7 discrete actions:
- **0**: Hold (không làm gì)
- **1**: Buy 25% của balance
- **2**: Buy 50% của balance
- **3**: Buy 100% của balance (all-in)
- **4**: Sell 25% của position
- **5**: Sell 50% của position
- **6**: Sell 100% của position (close all)

### Reward Function

Reward dựa trên:
- Portfolio value change (%)
- Profit percentage so với initial balance
- Penalty cho việc không giao dịch quá lâu
- Bonus cho profit dương

## 🎨 Visualization

Sau khi backtest, script tự động tạo các biểu đồ:

1. **Portfolio Value Over Time**: Theo dõi giá trị portfolio
2. **Action Distribution**: Phân bố các actions được thực hiện
3. **Trade Timeline**: Timeline của các giao dịch buy/sell

## 📊 Kết quả mẫu

```
==========================================
BACKTEST RESULTS
==========================================
Initial Balance: $10,000.00
Final Balance: $10,523.45
Profit: $523.45 (5.23%)
Total Trades: 47
==========================================

STRATEGY COMPARISON
==========================================
RL Agent:
  Return: 5.23%
  Trades: 47

Buy & Hold:
  Return: 3.12%
  Trades: 2

Outperformance: 2.11%
==========================================
```

## ⚙️ Tùy chỉnh

### Thay đổi symbols

```python
symbol_list = ["ADAUSDT", "SOLUSDT", "DOTUSDT"]
```

### Thay đổi interval

```python
interval = "5m"  # 5m, 15m, 1h, 4h, 1d
```

### Điều chỉnh hyperparameters

```python
trainer.train(
    total_timesteps=500000,     # Tăng để train lâu hơn
    learning_rate=1e-4,         # Learning rate thấp hơn
    n_steps=4096,               # Nhiều steps hơn
    batch_size=128,             # Batch size lớn hơn
    n_epochs=20                 # Nhiều epochs hơn
)
```

### Thay đổi commission và balance

```python
env = CryptoTradingEnv(
    df=df,
    initial_balance=50000.0,    # Vốn ban đầu
    commission=0.001,           # 0.1% phí
    max_position=0.8,           # Max 80% vốn
    window_size=100             # Window size lớn hơn
)
```

## 🐛 Troubleshooting

### Lỗi khi fetch data

```python
# Nếu Binance API bị limit, tăng delay
time.sleep(1)  # trong data_loader.py
```

### Out of memory

```python
# Giảm n_steps hoặc batch_size
n_steps=1024
batch_size=32
```

### Model không học

- Tăng learning rate
- Tăng entropy coefficient (ent_coef)
- Kiểm tra reward function
- Tăng số timesteps training

## 📝 Tips

1. **Data quality**: Đảm bảo dữ liệu không có gaps hoặc NaN
2. **Training time**: Ít nhất 200k timesteps để có kết quả tốt
3. **Multiple symbols**: Train trên nhiều symbols giúp model generalize tốt hơn
4. **Backtesting**: Luôn test trên dữ liệu out-of-sample
5. **Commission**: Đừng quên tính phí giao dịch thực tế (0.1-0.2%)

## 🔮 Phát triển thêm

- [ ] Thêm Stop-loss/Take-profit logic
- [ ] Position sizing động dựa trên volatility
- [ ] Multi-timeframe analysis
- [ ] Sentiment analysis từ news
- [ ] Live trading integration
- [ ] Risk management metrics (Sharpe, Sortino, Max Drawdown)

## 📚 Tài liệu tham khảo

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Binance API Documentation](https://binance-docs.github.io/apidocs/)

## ⚠️ Disclaimer

Code này chỉ cho mục đích giáo dục và nghiên cứu. Trading cryptocurrency có rủi ro cao. Không sử dụng với tiền thật mà không hiểu rõ code và rủi ro.

## 📄 License

MIT License
