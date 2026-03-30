import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from typing import Dict, List, Tuple, Optional

class CryptoTradingEnv(gym.Env):
    """
    Môi trường trading crypto cho Reinforcement Learning
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        commission: float = 0.001,  # 0.1% phí giao dịch
        max_position: float = 1.0,  # Tỷ lệ tối đa vốn có thể sử dụng
        window_size: int = 50,
    ):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_position = max_position
        self.window_size = window_size
        
        # Action space: [hold, buy_25%, buy_50%, buy_100%, sell_25%, sell_50%, sell_100%]
        self.action_space = spaces.Discrete(7)
        
        # Observation space: OHLCV + Technical indicators + Account info
        # window_size bars * 5 OHLCV + indicators + position info
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, 15),  # 15 features per timestep
            dtype=np.float32
        )
        
        # Tính toán indicators
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
        """Lấy observation cho timestep hiện tại"""
        start = max(0, self.current_step - self.window_size + 1)
        end = self.current_step + 1
        
        # Lấy dữ liệu window
        window_data = self.df.iloc[start:end].copy()
        
        # Nếu window chưa đủ, pad với zeros
        if len(window_data) < self.window_size:
            padding = pd.DataFrame(
                np.zeros((self.window_size - len(window_data), len(window_data.columns))),
                columns=window_data.columns
            )
            window_data = pd.concat([padding, window_data], ignore_index=True)
        
        # Normalize OHLCV
        current_price = self.df.iloc[self.current_step]['close']
        
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
        """Thực hiện action"""
        current_price = self.df.iloc[self.current_step]['close']
        
        # 0: Hold
        if action == 0:
            return
        
        # 1-3: Buy actions
        elif action in [1, 2, 3]:
            buy_ratios = {1: 0.25, 2: 0.5, 3: 1.0}
            ratio = buy_ratios[action]
            
            available_balance = self.balance * ratio * self.max_position
            cost = available_balance * (1 + self.commission)
            
            if cost <= self.balance:
                crypto_bought = available_balance / current_price
                self.crypto_held += crypto_bought
                self.balance -= cost
                self.total_trades += 1
                
                self.trade_history.append({
                    'step': self.current_step,
                    'type': 'BUY',
                    'price': current_price,
                    'amount': crypto_bought,
                    'cost': cost
                })
        
        # 4-6: Sell actions
        elif action in [4, 5, 6]:
            sell_ratios = {4: 0.25, 5: 0.5, 6: 1.0}
            ratio = sell_ratios[action]
            
            crypto_to_sell = self.crypto_held * ratio
            
            if crypto_to_sell > 0:
                revenue = crypto_to_sell * current_price * (1 - self.commission)
                self.balance += revenue
                self.crypto_held -= crypto_to_sell
                self.total_trades += 1
                
                self.trade_history.append({
                    'step': self.current_step,
                    'type': 'SELL',
                    'price': current_price,
                    'amount': crypto_to_sell,
                    'revenue': revenue
                })
    
    def _calculate_reward(self) -> float:
        """Tính toán reward"""
        current_price = self.df.iloc[self.current_step]['close']
        total_value = self.balance + self.crypto_held * current_price
        
        # Profit từ initial balance
        profit_pct = (total_value - self.initial_balance) / self.initial_balance
        
        # Reward dựa trên portfolio value change
        if len(self.balance_history) > 0:
            prev_value = self.balance_history[-1]
            value_change = (total_value - prev_value) / prev_value
            reward = value_change * 100  # Scale reward
        else:
            reward = 0
        
        # Penalty cho việc hold quá lâu không giao dịch
        if self.total_trades == 0 and self.current_step > 100:
            reward -= 0.1
        
        # Bonus cho profit dương
        if profit_pct > 0:
            reward += profit_pct * 10
        
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
        """Execute one step"""
        self._take_action(action)
        
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Get next observation
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
