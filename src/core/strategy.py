import numpy as np
import pandas as pd
import os
import torch


def save_directional_signals(df, symbol, time_interval, execution_window, signal_type, signal_threshold=0):
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
        (df['signal'].isin(['Buy', 'Sell']))
        # & (df['signal_strength'] > signal_threshold)
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


# def run_adaptive_backtest(
#         predictor_model,
#         dataset_1min,  # 1分钟数据的dataset（标准化、特征齐全）
#         raw_df_1min,  # 原始1分钟数据
#         signal_idx_1min,  # 信号在1min数据中的起始索引
#         model_input_seq_length,
#         execution_window,  # VWAP实际执行窗口（分钟数）
#         total_quantity,
#         trade_direction,
#         failsafe_ratio,
#         device,
#         vae_model
# ):
#     remaining_quantity = float(total_quantity)
#     results_log = []
#
#     # 仅在t=0时进行一次预测，获取整个窗口的预测序列，提高效率
#     predicted_ohlcv_entire_window = None
#
#     for t in range(execution_window):
#         order_quantity_for_this_minute = 0.0
#         logic_used = ""
#         predicted_close_for_this_minute = np.nan
#         predicted_volume_weight = np.nan
#         predicted_ohlcv_values = [np.nan] * 5  # 预测OHLCV，初始为NaN
#
#         # --- 边界条件1: 提前完成 ---
#         if remaining_quantity <= 1e-6:
#             logic_used = "已完成"
#             order_quantity_for_this_minute = 0.0
#             # 无需进行其他计算
#
#         # --- 边界条件2: 收盘冲刺 ---
#         elif t >= execution_window - 5:
#             logic_used = f"收盘冲刺 (r={failsafe_ratio})"
#             minutes_left = execution_window - t
#             if minutes_left == 1:
#                 order_quantity_for_this_minute = remaining_quantity
#             else:
#                 denominator = 1 - (failsafe_ratio ** minutes_left)
#                 if abs(denominator) < 1e-9:
#                     order_quantity_for_this_minute = remaining_quantity / minutes_left
#                 else:
#                     first_term = remaining_quantity * (1 - failsafe_ratio) / denominator
#                     order_quantity_for_this_minute = first_term
#             # 收盘冲刺阶段，不使用模型预测，因此预测值为空
#             predicted_volume_weight = np.nan
#             predicted_ohlcv_values = [np.nan] * 5
#
#         # --- 主要逻辑: 模型预测 ---
#         else:
#             logic_used = "模型预测"
#             input_start_idx = signal_idx_1min + t - model_input_seq_length
#             input_end_idx = signal_idx_1min + t
#
#             if input_start_idx >= 0:
#                 input_latent_seq_np = dataset_1min.normalized_latent[input_start_idx:input_end_idx]
#                 input_tensor = torch.FloatTensor(input_latent_seq_np).unsqueeze(0).to(device)
#
#                 latent_scaler = dataset_1min.latent_scaler if hasattr(dataset_1min, "latent_scaler") else None
#                 ohlcv_scaler = dataset_1min.ohlcv_scaler if hasattr(dataset_1min, "ohlcv_scaler") else None
#
#                 if vae_model is None or latent_scaler is None or ohlcv_scaler is None:
#                     print(f"vae_model, latent_scaler, ohlcv_scaler must be available for .generate()")
#                     exit(1)
#
#                 with torch.no_grad():
#                     # 仅在 t=0 时进行一次预测，获取整个窗口的预测序列
#                     if predicted_ohlcv_entire_window is None:
#                         gen_result = predictor_model.generate(
#                             input_tensor,
#                             vae_model=vae_model,
#                             ohlcv_scaler=ohlcv_scaler,
#                             latent_scaler=latent_scaler,
#                             steps=execution_window,
#                             device=device
#                         )
#                         outputs = gen_result['ohlcv'].squeeze(0).cpu().numpy()
#                         outputs = ohlcv_scaler.inverse_transform(outputs)
#                         predicted_ohlcv_entire_window = np.expm1(outputs)
#
#                     # 根据预测序列，计算当前分钟的订单量
#                     minutes_left_in_window = execution_window - t
#                     relevant_future_preds = predicted_ohlcv_entire_window[t:]
#                     predicted_volumes_in_window = relevant_future_preds[:, 4]  # volume index
#                     predicted_volume_for_now = predicted_volumes_in_window[0]
#
#                     sum_of_future_predicted_volumes = np.sum(predicted_volumes_in_window)
#
#                     # 新增：保存预测的OHLCV值
#                     predicted_ohlcv_values = relevant_future_preds[0, :5].tolist()
#
#                     if sum_of_future_predicted_volumes > 1e-6:
#                         weight_for_this_minute = predicted_volume_for_now / sum_of_future_predicted_volumes
#                         order_quantity_for_this_minute = remaining_quantity * weight_for_this_minute
#                         predicted_volume_weight = weight_for_this_minute  # 新增: 记录预测量占比
#                     else:
#                         order_quantity_for_this_minute = remaining_quantity / minutes_left_in_window
#                         predicted_volume_weight = 1 / minutes_left_in_window  # 新增: 记录预测量占比
#             else:
#                 # 预测所需数据不足时，采用均分策略
#                 order_quantity_for_this_minute = remaining_quantity / (execution_window - t)
#                 predicted_volume_weight = 1 / (execution_window - t)
#                 predicted_ohlcv_values = [np.nan] * 5
#
#         # --- 执行与记录 ---
#         final_order_quantity = max(0.0, min(remaining_quantity, order_quantity_for_this_minute))
#         current_minute_raw_data = raw_df_1min.iloc[signal_idx_1min + t]
#
#         remaining_quantity -= final_order_quantity
#
#         # 记录所需数据
#         results_log.append({
#             'timestamp': current_minute_raw_data['date'],
#             'predicted_open': predicted_ohlcv_values[0],
#             'predicted_high': predicted_ohlcv_values[1],
#             'predicted_low': predicted_ohlcv_values[2],
#             'predicted_close': predicted_ohlcv_values[3],
#             'predicted_volume': predicted_ohlcv_values[4],
#             'predicted_volume_weight': predicted_volume_weight,
#             'order_quantity': final_order_quantity,  # 这就是你说的“下单数量v”
#         })
#
#     # --- 后处理与指标计算 ---
#     results_df = pd.DataFrame(results_log)
#     if results_df.empty or results_df['order_quantity'].sum() < 1e-6:
#         return pd.DataFrame(), {}
#
#     # 原始的指标计算逻辑，用于生成metrics字典，这部分不变
#     total_actual_value_for_benchmark = (raw_df_1min.iloc[signal_idx_1min:signal_idx_1min + execution_window]['close'] *
#                                         raw_df_1min.iloc[signal_idx_1min:signal_idx_1min + execution_window][
#                                             'volume']).sum()
#     total_actual_volume_for_benchmark = raw_df_1min.iloc[signal_idx_1min:signal_idx_1min + execution_window][
#         'volume'].sum()
#     benchmark_vwap = total_actual_value_for_benchmark / total_actual_volume_for_benchmark if total_actual_volume_for_benchmark > 0 else 0
#     model_total_trade_value = (
#                 results_df['order_quantity'] * raw_df_1min.iloc[signal_idx_1min:signal_idx_1min + execution_window][
#             'close']).sum()
#     model_total_quantity_traded = results_df['order_quantity'].sum()
#     model_achieved_price = model_total_trade_value / model_total_quantity_traded if model_total_quantity_traded > 0 else 0
#     if trade_direction.lower() == 'buy':
#         slippage_per_share = benchmark_vwap - model_achieved_price
#     else:
#         slippage_per_share = model_achieved_price - benchmark_vwap
#     total_cost_savings = slippage_per_share * total_quantity
#     slippage_bps = (slippage_per_share / benchmark_vwap) * 10000 if benchmark_vwap > 0 else 0
#
#     metrics = {
#         "Benchmark VWAP": benchmark_vwap,
#         "Model Achieved Price": model_achieved_price,
#         "Slippage Reduction (BPS)": slippage_bps,
#         "Total Cost Savings": total_cost_savings
#     }
#
#     # 返回最终精简的DataFrame和指标
#     return results_df.set_index('timestamp'), metrics


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

    # 新增逻辑: 根据时间粒度定义收盘冲刺窗口
    closeout_window_map = {
        1: 1,
        5: 1,
        15: 5,
        60: 10
    }
    # 默认为5分钟，如果selected_interval_label不在映射中
    closeout_window = closeout_window_map.get(execution_window, 15)

    for t in range(execution_window):
        order_quantity_for_this_minute = 0.0
        logic_used = ""
        predicted_close_for_this_minute = np.nan
        predicted_volume_weight = np.nan  # 新增: 预测量占比
        predicted_ohlcv_values = [np.nan] * 5  # 新增: 预测OHLCV，初始为NaN

        # --- 边界条件1: 提前完成 ---
        if remaining_quantity <= 1e-6:
            logic_used = "已完成"
            order_quantity_for_this_minute = 0.0

        # --- 边界条件2: 收盘冲刺 ---
        elif t >= execution_window - closeout_window:
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
                    # 收盘冲刺阶段，不使用模型预测，因此预测值为空
            predicted_volume_weight = np.nan
            predicted_ohlcv_values = [np.nan] * 5
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

                    predicted_ohlcv_values = relevant_future_preds[0, :5].tolist()

                    if sum_of_future_predicted_volumes > 1e-6:
                        weight_for_this_minute = predicted_volume_for_now / sum_of_future_predicted_volumes
                        order_quantity_for_this_minute = remaining_quantity * weight_for_this_minute
                        predicted_volume_weight = weight_for_this_minute  # 新增: 记录预测量占比
                    else:
                        order_quantity_for_this_minute = remaining_quantity / minutes_left_in_window
                        predicted_volume_weight = 1 / minutes_left_in_window  # 新增: 记录预测量占比
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
            'predicted_open': predicted_ohlcv_values[0],  # 新增
            'predicted_high': predicted_ohlcv_values[1],  # 新增
            'predicted_low': predicted_ohlcv_values[2],  # 新增
            'predicted_close': predicted_ohlcv_values[3],  # 新增
            'predicted_volume': predicted_ohlcv_values[4],  # 新增
            'predicted_volume_weight': predicted_volume_weight,  # 新增
            'actual_volume': current_minute_raw_data['volume'],
            'predicted_price': predicted_close_for_this_minute,
            'execution_price': execution_price,
            # 'order_quantity': final_order_quantity,
            # 'trade_value': final_order_quantity * execution_price,
            # 'remaining_quantity': remaining_quantity
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
    metrics = {"naive VWAP": benchmark_vwap, "Adaptive VWAP Price": model_achieved_price,
               "Slippage Reduction (BPS)": slippage_bps, "Total Cost Savings": total_cost_savings}
    return results_df.set_index('timestamp'), metrics
