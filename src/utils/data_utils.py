import numpy as np
import pandas as pd
import streamlit as st
import torch
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


# --- 【关键修改 B】: 回测逻辑现在需要同时访问重采样数据和原始1分钟数据 ---
def run_adaptive_backtest(
        predictor_model,
        dataset_1min,  # 1分钟数据的dataset（标准化、特征齐全）
        raw_df_1min,  # 原始1分钟数据
        signal_idx_1min,  # 信号在1min数据中的起始索引
        model_input_seq_length,
        execution_window,  # VWAP实际执行窗口（分钟数）
        total_quantity,
        trade_direction,
        failsafe_ratio,
        device,
        vae_model
):
    remaining_quantity = float(total_quantity)
    results_log = []

    for t in range(execution_window):
        order_quantity_for_this_minute = 0.0
        logic_used = ""
        predicted_close_for_this_minute = np.nan

        # --- 边界条件1: 提前完成 ---
        if remaining_quantity <= 1e-6:
            logic_used = "已完成"
            order_quantity_for_this_minute = 0.0

        # --- 边界条件2: 收盘冲刺 ---
        elif t >= execution_window - 5:
            logic_used = f"收盘冲刺 (r={failsafe_ratio})"
            minutes_left = execution_window - t
            if minutes_left == 1:
                order_quantity_for_this_minute = remaining_quantity
            else:
                denominator = 1 - (failsafe_ratio ** minutes_left)
                if abs(denominator) < 1e-9:
                    order_quantity_for_this_minute = remaining_quantity / minutes_left
                else:
                    first_term = remaining_quantity * (1 - failsafe_ratio) / denominator
                    order_quantity_for_this_minute = first_term

        # --- 主要逻辑: 模型预测 ---
        else:
            logic_used = "模型预测"
            input_start_idx = signal_idx_1min + t - model_input_seq_length
            input_end_idx = signal_idx_1min + t

            if input_start_idx >= 0:
                input_latent_seq_np = dataset_1min.normalized_latent[input_start_idx:input_end_idx]
                input_tensor = torch.FloatTensor(input_latent_seq_np).unsqueeze(0).to(device)

                latent_scaler = None
                ohlcv_scaler = None
                if hasattr(dataset_1min, "latent_scaler"):
                    latent_scaler = dataset_1min.latent_scaler
                if hasattr(dataset_1min, "ohlcv_scaler"):
                    ohlcv_scaler = dataset_1min.ohlcv_scaler

                if vae_model is None or latent_scaler is None or ohlcv_scaler is None:
                    print(f"{vae_model}, {latent_scaler}, {ohlcv_scaler} must be available for .generate()")
                    exit(1)

                with torch.no_grad():
                    gen_result = predictor_model.generate(
                        input_tensor,
                        vae_model=vae_model,
                        ohlcv_scaler=ohlcv_scaler,
                        latent_scaler=latent_scaler,
                        steps=execution_window,
                        device=device
                    )
                    outputs = gen_result['ohlcv'].squeeze(0).cpu().numpy()
                    outputs = ohlcv_scaler.inverse_transform(outputs)
                    preds_raw = np.expm1(outputs)

                    minutes_left_in_window = execution_window - t
                    relevant_future_preds = preds_raw[t:]
                    predicted_volumes_in_window = relevant_future_preds[:, 4]  # volume index
                    predicted_volume_for_now = predicted_volumes_in_window[0]
                    sum_of_future_predicted_volumes = np.sum(predicted_volumes_in_window)
                    predicted_close_for_this_minute = relevant_future_preds[0, 3]  # close index

                    if sum_of_future_predicted_volumes > 1e-6:
                        weight_for_this_minute = predicted_volume_for_now / sum_of_future_predicted_volumes
                        order_quantity_for_this_minute = remaining_quantity * weight_for_this_minute
                    else:
                        order_quantity_for_this_minute = remaining_quantity / minutes_left_in_window
            else:
                order_quantity_for_this_minute = remaining_quantity / (execution_window - t)

        # --- 执行与记录 ---
        final_order_quantity = max(0.0, min(remaining_quantity, order_quantity_for_this_minute))
        current_minute_raw_data = raw_df_1min.iloc[signal_idx_1min + t]
        execution_price = (current_minute_raw_data['high'] + current_minute_raw_data['low'] + current_minute_raw_data[
            'close']) / 3.0

        # ；添加执行窗口开始和结束标志
        time_mark = ""
        if t == 0:  # t为0，代表循环开始，即执行窗口的开始
            time_mark = "start"
        elif t == execution_window - 1:  # t为execution_window-1，代表循环结束，即执行窗口的结束
            time_mark = "end"

        remaining_quantity -= final_order_quantity

        results_log.append({
            'timestamp': current_minute_raw_data['date'],
            'execution': time_mark,
            'logic_used': logic_used,
            'predicted_price': predicted_close_for_this_minute,
            'execution_price': execution_price,
            'actual_volume': current_minute_raw_data['volume'],
            'order_quantity': final_order_quantity,
            'trade_value': final_order_quantity * execution_price,
            'remaining_quantity': remaining_quantity
        })

    # --- 后处理与指标计算 ---
    results_df = pd.DataFrame(results_log)
    if results_df.empty or results_df['order_quantity'].sum() < 1e-6:
        return pd.DataFrame(), {}

    total_actual_value_for_benchmark = (results_df['execution_price'] * results_df['actual_volume']).sum()
    total_actual_volume_for_benchmark = results_df['actual_volume'].sum()
    benchmark_vwap = total_actual_value_for_benchmark / total_actual_volume_for_benchmark if total_actual_volume_for_benchmark > 0 else 0
    model_total_trade_value = results_df['trade_value'].sum()
    model_total_quantity_traded = results_df['order_quantity'].sum()
    model_achieved_price = model_total_trade_value / model_total_quantity_traded if model_total_quantity_traded > 0 else 0
    if trade_direction.lower() == 'buy':
        slippage_per_share = benchmark_vwap - model_achieved_price
    else:
        slippage_per_share = model_achieved_price - benchmark_vwap
    total_cost_savings = slippage_per_share * total_quantity
    slippage_bps = (slippage_per_share / benchmark_vwap) * 10000 if benchmark_vwap > 0 else 0
    results_df['cumulative_benchmark_value'] = (results_df['execution_price'] * results_df['actual_volume']).cumsum()
    results_df['cumulative_actual_volume'] = results_df['actual_volume'].cumsum()
    results_df['traditional_vwap_line'] = results_df['cumulative_benchmark_value'] / results_df[
        'cumulative_actual_volume']
    results_df['cumulative_model_value'] = results_df['trade_value'].cumsum()
    results_df['cumulative_model_volume'] = results_df['order_quantity'].cumsum()
    results_df['model_vwap_line'] = (
            results_df['cumulative_model_value'] / results_df['cumulative_model_volume']).replace([np.inf, -np.inf],
                                                                                                  np.nan).ffill()
    metrics = {"Benchmark VWAP": benchmark_vwap, "Model Achieved Price": model_achieved_price,
               "Slippage Reduction (BPS)": slippage_bps, "Total Cost Savings": total_cost_savings}
    return results_df.set_index('timestamp'), metrics


import random


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
