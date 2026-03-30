import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from typing import Dict, List, Tuple, Optional

class CryptoTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(
            self, 
            df: pd.DataFrame,
            initial_balance: float = 1000000.0,
            commission: float = 0.001,           # 0.1 % phi giao dich 
            max_position: float = 0.05,           # Ti le toi da von co the su dung 
            window_size = 50, 
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance 
        self.commission = commission 
        self.max_position = max_position
        self.window_size = window_size 

        # Action space: [hold, buy_25%, buy_50%, buy_100%, sell_25%, sell_50%, sell_100%]
        self.action_space = spaces.Discrete(9)

        # Observation space: OHLCV + Technical indicatoors + Account info 
        # Window_size bars * 5 OHLCV + indicators + position info 
        self.observation_space = spaces.Box(
            low = -np.inf,
            high = np.inf, 
            shape = (window_size, 15),  # 15 features 
            dtype = np.float32
        )

        # Tinh toan indicators 
        self._calculate_indicators()

        # State variables 
        self.current_step = 0 
        self.balance = initial_balance 
        self.crypto_held = 0
        self.total_trades = 0 
        self.total_profit = 0 

        # History tracking
        self.trade_history = []
        self.balance_history = []

    def _calculate_indicators(self):
        """Tính toán các chỉ báo kỹ thuật"""
        df = self.df.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # Fill NaN values
        df.fillna(0, inplace=True)
        
        self.df = df
    def _get_observation(self) -> np.ndarray:
        # "Lay observation cho tiemstamp hien tai"
        # "Nhin ve qua khu 30 gia + 1 gia hien tai la 31"
        start = max(0, self.current_step - self.window_size + 1)
        end = self.current_step + 1

        # "Lay du lieu window"
        window_data = self.df.iloc[start:end].copy()

        # Neu du lieu window khong du, pad voi zero 
        if len(window_data) < self.window_size:
            padding = pd.DataFrame(
                np.zeros((self.window_size - len(window_data), len(window_data.columns))),
                columns=window_data.columns
            )
            window_data = pd.concat([padding, window_data], ignore_index=True)

        # Normalize OHLCV 
        current_price = self.df.iloc[self.current_step]["close"]

        features = []
        for _, row in window_data.iterrows():
            feature_vector = [
                row['open'] / current_price,
                row['high'] / current_price,
                row['low'] / current_price,
                row['close'] / current_price,
                row['volume'] / (window_data['volume'].mean() + 1e-8),
                row['rsi'] / 100,
                row['macd'] / current_price,
                row['macd_signal'] / current_price,
                row['bb_upper'] / current_price,
                row['bb_middle'] / current_price,
                row['bb_lower'] / current_price,
                row['sma_20'] / current_price,
                row['ema_20'] / current_price,
                row['volume_sma'] / (window_data['volume'].mean() + 1e-8),
                self.crypto_held * current_price / (self.balance + self.crypto_held * current_price + 1e-8)
            ]
            features.append(feature_vector)
        return np.array(features, dtype=np.float32)
    
    def _take_action(self, action: int):
        """Thuc hien cac action"""
        current_price = self.df.iloc[self.current_step]['close']

        # Hold 
        if action == 0: 
            return 
        
        # 1-4: Buy actions 
        elif action in [1, 2, 3, 4]: 
            buy_ratios = {
                1: 0.25, 
                2: 0.5,
                3: 0.75,
                4: 1.0
            }
            ratio = buy_ratios[action]

            available_balance = self.balance * ratio * self.max_position
            cost = available_balance + (1 + self.commission)

            if cost <= self.balance:
                cryto_bought = available_balance / current_price 
                self.crypto_held += cryto_bought 
                self.balance -= cost 
                self.total_trades += 1 
                

                self.trade_history.append({
                    "step": self.current_step,
                    "number": cryto_bought,
                    "type": "Buy",
                    "price": current_price,
                    "cost": cost 
                })
        
        # 5-8: Sell actions 
        elif action in [5, 6, 7, 8]:
            sell_ratios = { 
                5: 0.25, 
                6: 0.5,
                7: 0.75, 
                8: 1.0
            }
            ratio = sell_ratios[action]

            crypto_to_sell = self.crypto_held * ratio 

            if crypto_to_sell > 0: 
                revenue = crypto_to_sell * current_price * (1 - self.commission)
                self.balance += revenue
                self.crypto_held -= crypto_to_sell
                self.total_trades += 1

                self.trade_history.append({
                    'step': self.current_step,
                    'number': crypto_to_sell,
                    'type': 'SELL',
                    'price': current_price,
                    'amount': crypto_to_sell,
                    'revenue': revenue
                })

    def _calculator_reward(self) -> float:
        current_price = self.df.iloc[self.current_step]['close']
        total_value = self.balance + self.crypto_held * current_price
        profit_pct = (total_value - self.initial_balance) / self.initial_balance

        # Reward chính dựa trên thay đổi giá trị
        if len(self.balance_history) > 0:
            prev_value = self.balance_history[-1]
            value_change = (total_value - prev_value) / prev_value
            reward = value_change * 50   # giảm bớt
        else:
            reward = 0

        # === PENALTY MẠNH ===
        # Phạt phí giao dịch
        if self.total_trades > 0:
            reward -= self.total_trades * 0.05   # phạt mỗi lệnh

        # Phạt hold quá lâu không trade
        if self.total_trades == 0 and self.current_step > 200:
            reward -= 0.2

        # Phạt drawdown lớn
        if profit_pct < -0.3:      # lỗ > 30%
            reward -= 5.0

        # Bonus nếu profit tốt
        if profit_pct > 0.1:
            reward += profit_pct * 15

        self.balance_history.append(total_value)
        return reward
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.total_trades = 0
        self.total_profit = 0
        self.trade_history = []
        self.balance_history = []
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Execute one step 
        self._take_action(action)

        self.current_step += 1 

        # Check if done 
        done = self.current_step >= len(self.df) - 1
        truncated = False

        # Calculate reward 
        reward = self._calculator_reward()

        # Get next obs 
        obs = self._get_observation()

        # Info 
        current_price = self.df.iloc[self.current_step]['close']
        total_value = self.balance + self.crypto_held * current_price

        info = {
            'step': self.current_step,
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'total_value': total_value,
            'profit': total_value - self.initial_balance,
            'profit_pct': ((total_value - self.initial_balance) / self.initial_balance) * 100,
            'total_trades': self.total_trades
        }
        
        return obs, reward, done, truncated, info
    def render(self):
        """Render environment state"""
        current_price = self.df.iloc[self.current_step]['close']
        total_value = self.balance + self.crypto_held * current_price
        profit = total_value - self.initial_balance
        profit_pct = (profit / self.initial_balance) * 100
        
        print(f"\n{'='*60}")
        print(f"Step: {self.current_step}/{len(self.df)}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Crypto Held: {self.crypto_held:.6f}")
        print(f"Total Value: ${total_value:.2f}")
        print(f"Profit: ${profit:.2f} ({profit_pct:.2f}%)")
        print(f"Total Trades: {self.total_trades}")
        print(f"{'='*60}\n")  


        
              





        





    
    
    


    