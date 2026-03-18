import gymnasium as gym 
import numpy as np 
from gymnasium import spaces 


class Forex_Trading_Env(gym.Env):
    def __init__(self, df, window_size=30, sl_options=None, tp_options=None):
        super(Forex_Trading_Env, self).__init__()

        # Store the dataframe containing prices and indicators 

        # Xử lý lại index = sắp xếp lại index từ 0 đến n lại từ đầu 
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)

        # Observation parameters - Co nghia la agent co the nhin duoc bao nhieu data tu qua khu 
        self.observation_size = window_size

        # No cho biet rang trang thai cua ban dang o dong bao nhieu, co nghia la ban da di bao nhieu step
        self.current_step = 0
        self.done = False 

        # Khoang cat lo, chot loi 
        self.sl_options = sl_options if sl_options else [60, 90, 120]
        self.tp_options = tp_options if tp_options else [60, 90, 120]

        # n_feature - kich thuoc input truoc khi dua vao mang neuron de hoc 
        self.n_feature = self.df.shape[1]

        # n_spread: phi giao dich moi co phieu 
        self.n_spread = 0.001

        # equity : so du vi
        self.equity = 10000

        # action_map 
        # --- (action, stop loss, take profit)
        self.action_map = [(None, None, None)]

        for direction in [0, 1]:
            for sl in sl_options:
                for tp in tp_options:
                    self.action_map.append((direction, sl, tp))

        # Return a window of these features as a 2D array 
        self.observation_space = spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape=(
                self.observation_size,
                self.n_feature
            ), dtype=np.float32
        )

        # The number of valid action 
        self.action_space = spaces.Discrete(len(self.action_map))


        # Track open positons if you want or one positon at a time 
        self.positions = []

        # For logging curve 
        self.equity_curve = []   
        self.last_trade_info = None # Track the last trade details
    
    def _get_observation(self):
        # Vấn đề xảy ra ở đây, agent phải quan sát được 30 giá trị ở trong quá khứ
        # Nếu bắt đầu vào ngày 0 đến ngày 28, ta không có đủ data để đưa vào input 
        start = max(self.current_step - self.observation_size, 0)
        obs_df = self.df.iloc[start:self.current_step]

        # Nếu giá trị không đủ 
        if len(obs_df) < self.observation_size:
            padding_rows = self.observation_size - len(obs_df)
            first_part = np.tile(obs_df.iloc[0].values, (padding_rows, 1))
            obs_array = np.concatenate([first_part, obs_df.values], axis=0)
        else:
            obs_array = obs_df.values
        
        return obs_array.astype(np.float32)
     
    def _calculate_reward(self, direction, sl_pips, tp_pips):
        if self.current_step > self.n_steps - 1:
            return 0
        # ---- Gia mua --- 
        entry_price = self.df.loc[self.current_step, "Close"]
        # --- Gia dinh moi ---
        next_high = self.df.loc[self.current_step + 1, "High"]
        # --- Gia thap moi ---
        next_low = self.df.loc[self.current_step + 1, "Low"]

        pip_value = 0.0001
        cost_pips = 1.8

        sl_price_distance = sl_pips * pip_value 
        tp_price_distance = tp_pips * pip_value 

        if direction == 1:  # Long
            take_profit = entry_price + tp_price_distance
            stop_loss = entry_price - sl_price_distance

            hit_tp = next_high >= take_profit
            hit_sl = next_low <= stop_loss
            
            if hit_tp and hit_sl:
                pnl = -sl_price_distance          # worst-case: hit SL trước
            elif hit_tp:
                pnl = tp_price_distance           # lời
            elif hit_sl:
                pnl = -sl_price_distance          # lỗ
            else:
                next_close = self.df.loc[self.current_step + 1, "Close"]
                pnl = next_close - entry_price    # SỬA: đúng dấu cho Long

        elif direction == -1:  # Short
            take_profit = entry_price - tp_price_distance
            stop_loss = entry_price + sl_price_distance

            hit_tp = next_low <= take_profit
            hit_sl = next_high >= stop_loss
            
            if hit_tp and hit_sl:
                pnl = -sl_price_distance          # worst-case
            elif hit_tp:
                pnl = tp_price_distance           # lời
            else:
                next_close = self.df.loc[self.current_step + 1, "Close"]
                pnl = entry_price - next_close    # đã đúng
    def step(self, action):
        direction, sl, tp = self.action_map[action]

        reward = 0.0
        entry_price = None 
        exit_price = None 

        if direction is not None: 
            entry_price = self.df.loc[self.current_step, "Close"]
            reward = self._calculate_reward(direction, sl, tp)

            if self.current_step < self.n_steps - 1:
                exit_price = self.df.loc[self.current_step + 1, "Close"]
            else:
                exit_price = entry_price 
        self.last_trade_info = {
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": reward / 10000.0
        }

        self.equity_curve.append(self.equity)

        terminated = self.current_state >= self.n_steps - 2
        truncated = False

        self.current_state += 1
        obs = self._get_observation()

        info = {
            "equity": self.equity,
            "last_trade": self.last_trade_info
        }

        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_state = self.observation_size
        self.equity = 10000.0
        self.equity_curve = []
        self.last_trade_info = None

        return self._get_observation(), {}

    
    def render(self, mode='human'):
        """
        Optinal: print or plot debug info.
        """
        print(f"Step: {self.current_state}, Equity: {self.equity}")




        
