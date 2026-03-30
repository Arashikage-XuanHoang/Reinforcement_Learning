import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from typing import List, Dict
import json
from datetime import datetime

from trading_env import CryptoTradingEnv
from data_loader import BinanceDataLoader, prepare_training_data


class TensorboardCallback(BaseCallback):
    """
    Custom callback để log thêm metrics vào tensorboard
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_profits = []
        
    def _on_step(self) -> bool:
        # Log additional metrics
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'profit_pct' in info:
                self.logger.record('trading/profit_pct', info['profit_pct'])
                self.logger.record('trading/total_value', info['total_value'])
                self.logger.record('trading/total_trades', info['total_trades'])
        
        return True


class TrainingLogger(BaseCallback):
    """
    Callback để log và lưu kết quả training
    """
    
    def __init__(self, log_dir: str, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_profits = []
        self.episode_trades = []
        
    def _on_step(self) -> bool:
        # Log khi episode kết thúc
        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1
            
            if len(self.locals.get('infos', [])) > 0:
                info = self.locals['infos'][0]
                
                self.episode_rewards.append(self.locals.get('rewards', [0])[0])
                self.episode_profits.append(info.get('profit_pct', 0))
                self.episode_trades.append(info.get('total_trades', 0))
                
                if self.episode_count % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_profit = np.mean(self.episode_profits[-10:])
                    avg_trades = np.mean(self.episode_trades[-10:])
                    
                    print(f"\n{'='*70}")
                    print(f"Episode {self.episode_count}")
                    print(f"{'='*70}")
                    print(f"Avg Reward (last 10): {avg_reward:.2f}")
                    print(f"Avg Profit % (last 10): {avg_profit:.2f}%")
                    print(f"Avg Trades (last 10): {avg_trades:.1f}")
                    print(f"{'='*70}\n")
        
        return True
    
    def _on_training_end(self) -> None:
        # Lưu training history
        history = {
            'episode_rewards': self.episode_rewards,
            'episode_profits': self.episode_profits,
            'episode_trades': self.episode_trades
        }
        
        with open(f"{self.log_dir}/training_history.json", 'w') as f:
            json.dump(history, f)
        
        # Plot training curves
        self._plot_training_curves()
    
    def _plot_training_curves(self):
        """Vẽ biểu đồ training progress"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Rewards
        axes[0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        axes[0].plot(pd.Series(self.episode_rewards).rolling(window=10).mean(), 
                    label='Moving Average (10)', linewidth=2)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Training Rewards')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Profits
        axes[1].plot(self.episode_profits, alpha=0.6, label='Episode Profit %')
        axes[1].plot(pd.Series(self.episode_profits).rolling(window=10).mean(), 
                    label='Moving Average (10)', linewidth=2)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Profit %')
        axes[1].set_title('Trading Profit %')
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Trades
        axes[2].plot(self.episode_trades, alpha=0.6, label='Episode Trades')
        axes[2].plot(pd.Series(self.episode_trades).rolling(window=10).mean(), 
                    label='Moving Average (10)', linewidth=2)
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Number of Trades')
        axes[2].set_title('Trading Activity')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/training_curves.png", dpi=150)
        print(f"\n✓ Training curves saved to {self.log_dir}/training_curves.png")


class CryptoRLTrainer:
    """
    Main trainer class cho RL trading bot
    """
    
    def __init__(
        self,
        symbol_list: List[str],
        interval: str = "1m",
        initial_balance: float = 10000.0,
        window_size: int = 50,
        model_dir: str = "models",
        log_dir: str = "logs"
    ):
        self.symbol_list = symbol_list
        self.interval = interval
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.model_dir = model_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        self.data_dict = None
        self.train_envs = None
        self.eval_envs = None
        self.model = None
        
    def load_data(self, days: int = 7, use_cached: bool = True):
        """Load hoặc fetch data"""
        print(f"\n{'='*70}")
        print(f"LOADING DATA")
        print(f"{'='*70}\n")
        
        # Try to load from cache first
        if use_cached:
            data_dict = {}
            all_cached = True
            
            for symbol in self.symbol_list:
                csv_files = [f for f in os.listdir("data") if f.startswith(symbol) and f.endswith('.csv')]
                if csv_files:
                    latest_file = max(csv_files)
                    filepath = f"data/{latest_file}"
                    print(f"Loading cached data: {filepath}")
                    loader = BinanceDataLoader()
                    data_dict[symbol] = loader.load_from_csv(filepath)
                else:
                    all_cached = False
                    break
            
            if all_cached:
                self.data_dict = data_dict
                print(f"\n✓ Loaded cached data for all symbols")
                return
        
        # Fetch new data
        print("Fetching fresh data from Binance...")
        self.data_dict = prepare_training_data(
            symbols=self.symbol_list,
            interval=self.interval,
            days=days,
            save_csv=True
        )
    
    def prepare_environments(self, train_split: float = 0.8):
        """Chuẩn bị training và evaluation environments"""
        print(f"\n{'='*70}")
        print(f"PREPARING ENVIRONMENTS")
        print(f"{'='*70}\n")
        
        train_envs = []
        eval_envs = []
        
        for symbol, df in self.data_dict.items():
            # Split data
            split_idx = int(len(df) * train_split)
            train_df = df.iloc[:split_idx].reset_index(drop=True)
            eval_df = df.iloc[split_idx:].reset_index(drop=True)
            
            print(f"{symbol}:")
            print(f"  Total: {len(df)} candles")
            print(f"  Train: {len(train_df)} candles")
            print(f"  Eval:  {len(eval_df)} candles")
            
            # Create environments
            train_env = CryptoTradingEnv(
                df=train_df,
                initial_balance=self.initial_balance,
                window_size=self.window_size
            )
            eval_env = CryptoTradingEnv(
                df=eval_df,
                initial_balance=self.initial_balance,
                window_size=self.window_size
            )
            
            # Wrap with Monitor
            train_env = Monitor(train_env)
            eval_env = Monitor(eval_env)
            
            train_envs.append(train_env)
            eval_envs.append(eval_env)
        
        # Vectorize environments
        self.train_envs = DummyVecEnv([lambda env=env: env for env in train_envs])
        self.eval_envs = DummyVecEnv([lambda env=env: env for env in eval_envs])
        
        # Normalize observations
        self.train_envs = VecNormalize(self.train_envs, norm_obs=True, norm_reward=True)
        self.eval_envs = VecNormalize(self.eval_envs, norm_obs=True, norm_reward=False, training=False)
        
        print(f"\n✓ Created {len(train_envs)} training environments")
        print(f"✓ Created {len(eval_envs)} evaluation environments")
    
    def train(
        self,
        total_timesteps: int = 100000,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        save_freq: int = 10000
    ):
        """Train the RL model"""
        print(f"\n{'='*70}")
        print(f"TRAINING RL MODEL")
        print(f"{'='*70}")
        print(f"Algorithm: PPO")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Learning rate: {learning_rate}")
        print(f"N steps: {n_steps}")
        print(f"Batch size: {batch_size}")
        print(f"N epochs: {n_epochs}")
        print(f"{'='*70}\n")
        
        # Create model
        self.model = PPO(
            "MlpPolicy",
            self.train_envs,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=f"{self.log_dir}/tensorboard"
        )
        
        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=f"{self.model_dir}/checkpoints",
            name_prefix="ppo_crypto"
        )
        
        training_logger = TrainingLogger(log_dir=self.log_dir)
        tensorboard_callback = TensorboardCallback()
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, training_logger, tensorboard_callback],
            progress_bar=True
        )
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{self.model_dir}/ppo_crypto_final_{timestamp}"
        self.model.save(model_path)
        self.train_envs.save(f"{model_path}_vecnormalize.pkl")
        
        print(f"\n✓ Model saved to {model_path}")
    
    def evaluate(self, n_episodes: int = 10):
        """Evaluate the trained model"""
        print(f"\n{'='*70}")
        print(f"EVALUATING MODEL")
        print(f"{'='*70}\n")
        
        if self.model is None:
            print("Error: No model trained yet!")
            return
        
        all_profits = []
        all_trades = []
        all_returns = []
        
        for episode in range(n_episodes):
            obs = self.eval_envs.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_envs.step(action)
                episode_reward += reward[0]
                
                if done[0]:
                    if len(info) > 0 and 'profit_pct' in info[0]:
                        profit = info[0]['profit_pct']
                        trades = info[0]['total_trades']
                        all_profits.append(profit)
                        all_trades.append(trades)
                        all_returns.append(episode_reward)
                        
                        print(f"Episode {episode + 1}/{n_episodes}:")
                        print(f"  Profit: {profit:.2f}%")
                        print(f"  Trades: {trades}")
                        print(f"  Return: {episode_reward:.2f}")
        
        # Summary statistics
        print(f"\n{'='*70}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"Episodes: {n_episodes}")
        print(f"Avg Profit: {np.mean(all_profits):.2f}% (±{np.std(all_profits):.2f}%)")
        print(f"Max Profit: {np.max(all_profits):.2f}%")
        print(f"Min Profit: {np.min(all_profits):.2f}%")
        print(f"Avg Trades: {np.mean(all_trades):.1f}")
        print(f"Win Rate: {(np.array(all_profits) > 0).sum() / len(all_profits) * 100:.1f}%")
        print(f"{'='*70}\n")
        
        return {
            'profits': all_profits,
            'trades': all_trades,
            'returns': all_returns
        }


def main():
    """Main training script"""
    
    # Configuration
    symbol_list = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "LTCUSDT", "XRPUSDT"]
    interval = "1m"
    days = 7  # Số ngày dữ liệu
    
    # Training parameters
    total_timesteps = 200000  # Tăng lên cho training tốt hơn
    learning_rate = 3e-4
    
    print(f"\n{'='*70}")
    print(f"CRYPTO TRADING RL TRAINING")
    print(f"{'='*70}")
    print(f"Symbols: {', '.join(symbol_list)}")
    print(f"Interval: {interval}")
    print(f"Data days: {days}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"{'='*70}\n")
    
    # Initialize trainer
    trainer = CryptoRLTrainer(
        symbol_list=symbol_list,
        interval=interval,
        initial_balance=10000.0,
        window_size=50
    )
    
    # Load data
    trainer.load_data(days=days, use_cached=True)
    
    # Prepare environments
    trainer.prepare_environments(train_split=0.8)
    
    # Train
    trainer.train(
        total_timesteps=total_timesteps,
        learning_rate=learning_rate,
        save_freq=10000
    )
    
    # Evaluate
    results = trainer.evaluate(n_episodes=10)
    
    print("\n✓ Training and evaluation complete!")
    print("\nTo view training progress, run:")
    print("  tensorboard --logdir logs/tensorboard")


if __name__ == "__main__":
    main()
