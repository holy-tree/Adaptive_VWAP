import pandas as pd
import os

def save_directional_signals(df, symbol, time_interval, execution_window, signal_type, signal_threshold=0.6):
    """
    根据给定的阈值和信号类型，筛选并保存交易信号到CSV文件。
    保存 date buy/sell

    Args:
        df (pd.DataFrame): 包含'date', 'signal', 'signal_strength'等列的数据。
        symbol (str): 交易合约符号。
        time_interval (str): 数据时间周期。
        execution_window (int): 回测执行窗口。
        signal_type (str): 信号类型，例如 'bollinger', 'macd' 等。
        signal_threshold (float): 信号强度阈值，信号强度低于此值的将被忽略。
    """
    # 定义信号保存的文件夹
    signals_dir = f"../../results/signals/{signal_type}_signals"
    os.makedirs(signals_dir, exist_ok=True)

    # 基于阈值和信号类型筛选数据
    filtered_signals_df = df[
        (df['signal'].isin(['Buy', 'Sell'])) &
        (df['signal_strength'] > signal_threshold)
    ].copy()

    # 格式化输出 DataFrame
    if not filtered_signals_df.empty:
        # 构造文件名，将所有关键信息都包含在内
        signal_filename = f"{signal_type}_signals_{time_interval}_{symbol}_win{execution_window}.csv"
        signal_filepath = os.path.join(signals_dir, signal_filename)

        # 保存 CSV 文件，只包含 'date' 和 'signal' 列
        filtered_signals_df[['date', 'signal']].to_csv(signal_filepath, index=False)
        print(f"Signals for {symbol} saved to {signal_filepath}")
    else:
        print(f"Warning: No valid signals found for {symbol} at {time_interval} with window={execution_window}.")