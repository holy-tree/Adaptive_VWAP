import pandas_ta as ta
import pandas as pd

def generate_bollinger_signals_with_strength(df, length=20, std=2.0, stl_param=5.0, n_param=6.0):
    """
    基于布林带的双向交易信号生成策略，并计算信号强度。
    Args:
        df (pd.DataFrame): 包含'close', 'high', 'low'列的数据。
        ... (其他参数同上) ...
    Returns:
        pd.DataFrame: 添加'signal'和'signal_strength'列后的数据。
    """
    if df.empty:
        df['signal'] = []
        df['signal_strength'] = []
        return df

    bbands = ta.bbands(df['close'], length=length, std=std)
    df['bb_upper'] = bbands[f'BBU_{length}_{std}']
    df['bb_middle'] = bbands[f'BBM_{length}_{std}']
    df['bb_lower'] = bbands[f'BBL_{length}_{std}']
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=length)

    signals = ['Hold'] * len(df)
    strengths = [0.0] * len(df)
    position = 0
    bkhigh = 0.0
    sklow = float('inf')

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # --- 止损或止盈逻辑 ---
        if position == 1:
            stop_loss_atr = bkhigh - n_param * row['atr']
            stop_loss_stl = prev_row['close'] * (1 - 0.01 * stl_param)
            if row['close'] <= stop_loss_atr or prev_row['low'] < stop_loss_stl:
                signals[i] = 'Close_Buy'
                position = 0
                continue
            bkhigh = max(bkhigh, row['high'])

        elif position == -1:
            stop_loss_atr = sklow + n_param * row['atr']
            stop_loss_stl = prev_row['close'] * (1 + 0.01 * stl_param)
            if row['close'] >= stop_loss_atr or prev_row['high'] > stop_loss_stl:
                signals[i] = 'Close_Sell'
                position = 0
                continue
            sklow = min(sklow, row['low'])

        # --- 入场逻辑 ---
        if position == 0:
            # 开多：由下轨下穿反转向上
            if prev_row['close'] < prev_row['bb_lower'] and row['close'] > row['bb_lower']:
                signals[i] = 'Buy'
                position = 1
                bkhigh = row['high']
                # 计算信号强度：价格越偏离中轨，信号越强
                strength = (row['close'] - row['bb_lower']) / (row['bb_middle'] - row['bb_lower'])
                strengths[i] = max(0.0, min(1.0, strength))
            # 开空：由上轨上穿反转向下
            elif prev_row['close'] > prev_row['bb_upper'] and row['close'] < row['bb_upper']:
                signals[i] = 'Sell'
                position = -1
                sklow = row['low']
                # 计算信号强度：价格越偏离中轨，信号越强
                strength = (row['bb_upper'] - row['close']) / (row['bb_upper'] - row['bb_middle'])
                strengths[i] = max(0.0, min(1.0, strength))

    df['signal'] = signals
    df['signal_strength'] = strengths
    return df