import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import seaborn as sns
from typing import List, Dict
import os

from trading_env import CryptoTradingEnv
from data_loader import BinanceDataLoader


class TradingBot:
    """
    Trading bot sử dụng trained RL model
    """
    
    def __init__(self, model_path: str, vecnormalize_path: str = None):
        """
        Load trained model
        
        Args:
            model_path: Path đến model file
            vecnormalize_path: Path đến VecNormalize stats
        """
        print(f"Loading model from {model_path}...")
        self.model = PPO.load(model_path)
        
        self.vecnormalize = None
        if vecnormalize_path and os.path.exists(vecnormalize_path):
            print(f"Loading VecNormalize stats from {vecnormalize_path}...")
            # Will be applied to env later
            self.vecnormalize_path = vecnormalize_path
        
        print("✓ Model loaded successfully!")
    
    def backtest(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        render_freq: int = 0
    ) -> Dict:
        """
        Backtest model trên dữ liệu
        
        Args:
            df: DataFrame chứa OHLCV data
            initial_balance: Số vốn ban đầu
            render_freq: Tần suất render (0 = không render)
        
        Returns:
            Dictionary chứa kết quả backtest
        """
        print(f"\n{'='*70}")
        print(f"BACKTESTING")
        print(f"{'='*70}")
        print(f"Data points: {len(df)}")
        print(f"Initial balance: ${initial_balance:,.2f}")
        print(f"{'='*70}\n")
        
        # Create environment
        env = CryptoTradingEnv(
            df=df,
            initial_balance=initial_balance,
            window_size=50
        )
        env = Monitor(env)
        
        # Wrap with VecEnv and VecNormalize if available
        vec_env = DummyVecEnv([lambda: env])
        
        if hasattr(self, 'vecnormalize_path'):
            vec_env = VecNormalize.load(self.vecnormalize_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
        
        # Run backtest
        obs = vec_env.reset()
        done = False
        
        portfolio_values = []
        actions_taken = []
        trade_history = []
        
        step = 0
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            
            if render_freq > 0 and step % render_freq == 0:
                env.render()
            
            # Track metrics
            if len(info) > 0:
                portfolio_values.append(info[0].get('total_value', initial_balance))
                actions_taken.append(action[0])
            
            step += 1
        
        # Get final results
        final_info = info[0] if len(info) > 0 else {}
        
        results = {
            'initial_balance': initial_balance,
            'final_balance': final_info.get('total_value', initial_balance),
            'profit': final_info.get('profit', 0),
            'profit_pct': final_info.get('profit_pct', 0),
            'total_trades': final_info.get('total_trades', 0),
            'portfolio_values': portfolio_values,
            'actions_taken': actions_taken,
            'trade_history': env.trade_history
        }
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"BACKTEST RESULTS")
        print(f"{'='*70}")
        print(f"Initial Balance: ${results['initial_balance']:,.2f}")
        print(f"Final Balance: ${results['final_balance']:,.2f}")
        print(f"Profit: ${results['profit']:,.2f} ({results['profit_pct']:.2f}%)")
        print(f"Total Trades: {results['total_trades']}")
        print(f"{'='*70}\n")
        
        return results
    
    def plot_backtest_results(self, results: Dict, save_path: str = None):
        """
        Vẽ biểu đồ kết quả backtest
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Portfolio value over time
        portfolio_values = results['portfolio_values']
        axes[0].plot(portfolio_values, linewidth=2, color='#2E86AB')
        axes[0].axhline(y=results['initial_balance'], color='r', linestyle='--', 
                       label='Initial Balance', alpha=0.7)
        axes[0].fill_between(range(len(portfolio_values)), 
                            results['initial_balance'], portfolio_values,
                            alpha=0.3, color='green' if results['profit'] > 0 else 'red')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].set_title(f'Portfolio Value Over Time (Final: ${results["final_balance"]:,.2f}, '
                         f'Profit: {results["profit_pct"]:.2f}%)', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Actions distribution
        actions_taken = results['actions_taken']
        action_labels = ['Hold', 'Buy 25%', 'Buy 50%', 'Buy 100%', 
                        'Sell 25%', 'Sell 50%', 'Sell 100%']
        action_counts = [actions_taken.count(i) for i in range(7)]
        
        colors = ['gray', 'lightgreen', 'green', 'darkgreen', 
                 'lightcoral', 'red', 'darkred']
        axes[1].bar(action_labels, action_counts, color=colors, alpha=0.7)
        axes[1].set_xlabel('Action')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Action Distribution', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Trade timeline
        if results['trade_history']:
            trade_df = pd.DataFrame(results['trade_history'])
            
            buy_trades = trade_df[trade_df['type'] == 'BUY']
            sell_trades = trade_df[trade_df['type'] == 'SELL']
            
            axes[2].scatter(buy_trades['step'], buy_trades['price'], 
                          marker='^', s=100, c='green', alpha=0.7, label='Buy')
            axes[2].scatter(sell_trades['step'], sell_trades['price'], 
                          marker='v', s=100, c='red', alpha=0.7, label='Sell')
            
            axes[2].set_xlabel('Time Step')
            axes[2].set_ylabel('Price ($)')
            axes[2].set_title(f'Trade Timeline (Total: {results["total_trades"]} trades)', 
                            fontsize=12, fontweight='bold')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'No trades executed', 
                        ha='center', va='center', fontsize=14)
            axes[2].set_title('Trade Timeline', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        else:
            plt.show()
    
    def compare_with_buy_and_hold(self, df: pd.DataFrame, results: Dict):
        """
        So sánh với chiến lược Buy and Hold
        """
        initial_price = df.iloc[50]['close']  # Window size = 50
        final_price = df.iloc[-1]['close']
        
        buy_hold_return = ((final_price - initial_price) / initial_price) * 100
        
        print(f"\n{'='*70}")
        print(f"STRATEGY COMPARISON")
        print(f"{'='*70}")
        print(f"RL Agent:")
        print(f"  Return: {results['profit_pct']:.2f}%")
        print(f"  Trades: {results['total_trades']}")
        print(f"\nBuy & Hold:")
        print(f"  Return: {buy_hold_return:.2f}%")
        print(f"  Trades: 2 (1 buy, 1 sell)")
        print(f"\nOutperformance: {results['profit_pct'] - buy_hold_return:.2f}%")
        print(f"{'='*70}\n")


def test_on_symbol(
    model_path: str,
    symbol: str,
    days: int = 3,
    vecnormalize_path: str = None
):
    """
    Test model trên một symbol cụ thể
    """
    print(f"\n{'='*70}")
    print(f"TESTING MODEL ON {symbol}")
    print(f"{'='*70}\n")
    
    # Load data
    loader = BinanceDataLoader()
    
    # Try to load from cache first
    csv_files = [f for f in os.listdir("data") if f.startswith(symbol) and f.endswith('.csv')]
    
    if csv_files:
        latest_file = max(csv_files)
        filepath = f"data/{latest_file}"
        print(f"Loading data from {filepath}")
        df = loader.load_from_csv(filepath)
    else:
        print(f"Fetching fresh data for {symbol}...")
        df = loader.fetch_historical_data(symbol=symbol, interval="1m", days=days)
        if df.empty:
            print(f"Failed to load data for {symbol}")
            return None
    
    # Initialize bot
    bot = TradingBot(model_path=model_path, vecnormalize_path=vecnormalize_path)
    
    # Backtest
    results = bot.backtest(df=df, initial_balance=10000.0)
    
    # Plot results
    bot.plot_backtest_results(results, save_path=f"results_{symbol}_backtest.png")
    
    # Compare with buy and hold
    bot.compare_with_buy_and_hold(df, results)
    
    return results


def test_multiple_symbols(
    model_path: str,
    symbol_list: List[str],
    vecnormalize_path: str = None
):
    """
    Test model trên nhiều symbols
    """
    all_results = {}
    
    for symbol in symbol_list:
        results = test_on_symbol(
            model_path=model_path,
            symbol=symbol,
            vecnormalize_path=vecnormalize_path
        )
        if results:
            all_results[symbol] = results
    
    # Summary
    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}")
    
    for symbol, results in all_results.items():
        print(f"\n{symbol}:")
        print(f"  Profit: {results['profit_pct']:.2f}%")
        print(f"  Trades: {results['total_trades']}")
    
    avg_profit = np.mean([r['profit_pct'] for r in all_results.values()])
    avg_trades = np.mean([r['total_trades'] for r in all_results.values()])
    
    print(f"\nAverage across all symbols:")
    print(f"  Profit: {avg_profit:.2f}%")
    print(f"  Trades: {avg_trades:.1f}")
    print(f"{'='*70}\n")
    
    return all_results


if __name__ == "__main__":
    # Example usage
    model_path = "models/ppo_crypto_final_20250326_120000.zip"  # Update with your model path
    vecnormalize_path = "models/ppo_crypto_final_20250326_120000_vecnormalize.pkl"
    
    symbol_list = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "LTCUSDT", "XRPUSDT"]
    
    # Test on single symbol
    # test_on_symbol(model_path, "BTCUSDT", vecnormalize_path=vecnormalize_path)
    
    # Test on multiple symbols
    # results = test_multiple_symbols(model_path, symbol_list, vecnormalize_path=vecnormalize_path)
    
    print("\nTo test the model, uncomment the test functions above and provide the correct model path.")
