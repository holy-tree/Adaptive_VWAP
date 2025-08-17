import numpy as np
import pandas as pd
import streamlit as st
import torch
import random
from models.src.dataloader_setup import FinancialDataset


@st.cache_data
def load_financial_data(csv_path, seq_length, latent_dim):
    # 这个函数现在主要用于1分钟原始数据的Dataset构建
    dataset = FinancialDataset(latent_csv_path=csv_path, seq_length=seq_length, latent_dim=latent_dim)
    return dataset


@st.cache_data
def resample_data(df, interval):
    if interval == '1min':
        return df.copy()

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    latent_cols = [col for col in df.columns if 'latent_dim' in col]

    aggregation_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'amount': 'sum',
        'position': 'last'
    }

    for col in latent_cols:
        aggregation_rules[col] = 'mean'

    resampled_df = df.resample(interval).agg(aggregation_rules)
    resampled_df = resampled_df.dropna(subset=['open'])
    return resampled_df.reset_index()


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def get_signal_index(raw_df, signal_time):
    """
    在原始1分钟数据DataFrame中查找第一个信号时间的索引。
    如果没有精确匹配，则返回最接近且之前的那个时间点的索引。

    参数:
        raw_df (pd.DataFrame): 包含'date'列的原始1分钟数据DataFrame。
        signal_time (pd.Timestamp或datetime-like): 第一个信号的时间戳。

    返回:
        int: 对应数据点的整数索引。
    """
    # 1. 将信号时间转换为Pandas时间戳，以确保比较格式一致
    signal_time = pd.to_datetime(signal_time)

    # 2. 使用布尔掩码找到精确匹配的行
    exact_match_mask = raw_df['date'] == signal_time

    # 3. 检查是否存在精确匹配
    if exact_match_mask.any():
        # 如果找到，获取第一个精确匹配的索引
        # 使用 idxmax() 比过滤整个DataFrame更高效
        return exact_match_mask.idxmax()
    else:
        # 4. 如果没有精确匹配，则查找最接近且之前的那个时间点
        # 此掩码找到所有小于或等于信号时间的日期
        preceding_mask = raw_df['date'] <= signal_time

        # 检查是否有任何之前的数据
        if not preceding_mask.any():
            print("警告: 没有找到之前的数据点。返回索引0。")
            return 0

        # 找到最后一个（最近的）之前的那个点的索引
        # 在反转后的Series上使用 idxmax() 是找到最后一个True值的有效方法
        return preceding_mask.iloc[::-1].idxmax()


def get_and_validate_signals(df, direction_filter, selected_date):
    """
    筛选指定日期和方向的信号，并返回第一个信号。
    如果信号为空，则返回None。

    参数:
        df (pd.DataFrame): 包含交易信号的DataFrame。
        direction_filter (str): 交易方向 ('Buy' 或 'Sell')。
        selected_date (datetime.date): 选定的日期。

    返回:
        pd.Series or None: 第一个信号的Series，如果无信号则为None。
    """
    selected_date_df = df[df['date'].dt.date == selected_date].copy()
    entry_signals = selected_date_df[selected_date_df['signal'] == direction_filter]

    if entry_signals.empty:
        print(f"{selected_date} 无{direction_filter}信号，跳过。")
        return None

    return entry_signals.iloc[0]