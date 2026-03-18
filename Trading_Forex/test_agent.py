from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

from indicators import load_and_preprocess_data 
from trading_env import ForexTradingEnv

def main():
    # 1. Load new test data
    test_df = load_and_preprocess_data("/mnt/0498342198341420/UIT_DS_Third_Year_2025_2026/Reinforcement_Learning/Trading_Forex/data/test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv")

    # Create new same  environment 
    test_env = ForexTradingEnv(
        df=test_df,
        window_size=30,    # same as training 
        sl_options=[30, 60, 80],
        tp_options=[30, 60, 80]
    )

    # Wrap in a DummyVecEnv (required by stable_baselines for parallelization)
    vec_test_env = DummyVecEnv([lambda: test_env])

    # Define RL model (PPO)
    model = PPO.load("model_eurusd", env=vec_test_env) 




    # Initialize logs
    obs = vec_test_env.reset()
    done = False

    equity_curve = []

    # For trade tracking 
    trade_history = []
    trade_id = 1

    while True:
        action, _states = model.predict(obs, deterministic = True)
        obs, rewards, done, info = vec_test_env.step(action)

        # Collect equity from the unwrapped environment
        # Because we have a DummyVecEnv, we can access env_method to get the attribute
        current_equity = vec_test_env.get_attr("equity")[0]
        equity_curve.append(current_equity)

        if done[0]:
            break 
    # Plot the final equity curve 
    plt.figure(figsize=(15, 5))
    plt.plot(equity_curve, label="Equity")
    plt.title("Equity curve during Evaluation")
    plt.xlabel("Time Steps")
    plt.ylabel("Equity")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

