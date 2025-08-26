import os
import logging
import pandas as pd
import torch
import argparse

from generate_signal import generate_bollinger_signals
from src.core.strategy import save_directional_signals, run_adaptive_backtest
from src.utils.data_utils import resample_data, set_seed, load_financial_data, get_signal_index
from src.utils.model_utils import load_predictor_model, load_vae_model

SEQ_LENGTH = 345

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义周期与可选execution_window的映射
interval_to_windows = {
    '1min': [1],
    '5min': [1, 5],
    '15min': [1, 5, 15],
    '30min': [1, 5, 15, 30],
    '60min': [1, 5, 15, 30, 60],
    'D': [1, 5, 15, 30, 60, SEQ_LENGTH]  # 假设日线SEQ_LENGTH分钟
}

time_interval_map = {
    '1分钟': '1min', '5分钟': '5min', '15分钟': '15min',
    '30分钟': '30min', '1小时': '60min', '日线': 'D'
}

logging.basicConfig(level=logging.DEBUG)  # 默认 INFO，不会显示 DEBUG
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    for trade_direction in ["buy", "sell"]:
        for model_name in ["our-transformer", "rnn", "lstm"]:
            # for symbol in ["rb9999", "i9999", "cu9999", "ni9999", "sc9999", "pg9999", "y9999", "ag9999", "m9999",
            #                "c9999", "TA9999", "UR9999", "OI9999", "au9999", "IH9999", "T9999", "CF9999", "AP9999"]:
            for symbol in ["sc9999"]:
                set_seed(42)
                # 参数设置
                parser = argparse.ArgumentParser(description="VWAP批量回测参数")
                parser.add_argument('--data_file', type=str, default=f'../../data/latent_features/{symbol}_1min_data.csv')
                parser.add_argument('--vae_model_path', type=str, default=f'../../models/trained/VAE_models/{symbol}_tdist_vae_model.pth')
                parser.add_argument('--predictor_model_path', type=str,
                                    default=f'../../models/trained/predictor_models/{symbol}_{model_name}_predictor_model.pth')
                parser.add_argument('--total_quantity', type=float, default=10000)
                parser.add_argument('--failsafe_ratio', type=float, default=0.75)
                parser.add_argument('--execution_window', type=int, default=30)
                parser.add_argument('--length_param', type=int, default=20)
                parser.add_argument('--std_param', type=float, default=2.0)
                parser.add_argument('--stl_param', type=float, default=5.0)
                parser.add_argument('--n_param', type=float, default=6.0)
                parser.add_argument('--model_input_seq_length', type=int, default=345)
                parser.add_argument('--latent_dim', type=int, default=16)
                parser.add_argument('--ohlcv_dim', type=int, default=5)
                # 检测交易信号的频率buy sell hold
                parser.add_argument('--intervals', type=str, default='1min,5min,15min,30min,60min,D',
                                    help='逗号分隔的周期列表，如1min,5min,15min,30min,60min,D')
                parser.add_argument('--start_date', type=str, required=False, default="2025-06-04",
                                    help='回测开始日期，格式: YYYY-MM-DD')
                parser.add_argument('--end_date', type=str, required=False, default="2025-06-06",
                                    help='回测结束日期，格式: YYYY-MM-DD')

                args = parser.parse_args()

                # 解析周期映射
                intervals = args.intervals.split(',')
                time_interval_map = {k: v for k, v in time_interval_map.items() if v in intervals}

                # 解析参数
                data_file = args.data_file
                vae_model_path = args.vae_model_path
                predictor_model_path = None
                if model_name != "our-transformer":
                    predictor_model_path = args.predictor_model_path
                else:
                    predictor_model_path = f"../../models/trained/predictor_models/{symbol}_predictor_model.pth"
                total_quantity = args.total_quantity
                failsafe_ratio = args.failsafe_ratio
                execution_window = args.execution_window
                length_param = args.length_param
                std_param = args.std_param
                stl_param = args.stl_param
                n_param = args.n_param
                model_input_seq_length = args.model_input_seq_length
                latent_dim = args.latent_dim
                ohlcv_dim = args.ohlcv_dim

                # 加载模型
                predictor_model = load_predictor_model(model_name, predictor_model_path, device=device)
                vae_model = load_vae_model(vae_model_path, ohlcv_dim, latent_dim, 64, 5.0, device)

                # 加载1分钟数据集
                dataset_1min = load_financial_data(data_file, model_input_seq_length, latent_dim)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"---\n**分析开始** | 设备: `{device}`")

                # 开始遍历执行窗口
                for selected_interval_label, time_interval in time_interval_map.items():
                    try:
                        # 对原始1min数据进行重采样
                        logger.info(f"[{selected_interval_label}] 1. 加载并重采样数据...")
                        raw_df_1min = pd.read_csv(data_file, parse_dates=['date'])
                        resampled_df = resample_data(raw_df_1min.copy(), time_interval)

                        # STEP 1:
                        # STEP 1.1 根据信号生成器，生成buy/sell,只保留强度超过阈值的信号
                        bollinger_params = {"length": length_param, "std": std_param, "stl_param": stl_param,
                                            "n_param": n_param}
                        # 使用 resampled_df 作为输入，并确保它有时间索引
                        signals_df = resampled_df.copy()
                        bollinger_signal_df = generate_bollinger_signals(signals_df, **bollinger_params)

                        logger.debug(f"生成的Bollinger信号数: {len(bollinger_signal_df)}")
                        logger.debug(f"信号示例:\n{bollinger_signal_df.head()}")

                        allowed_windows = interval_to_windows.get(time_interval, [1])
                        # 基于阈值和信号类型筛选数据
                        for execution_window in allowed_windows:
                            save_directional_signals(
                                df=bollinger_signal_df,
                                symbol=symbol,
                                time_interval=time_interval,
                                execution_window=execution_window,
                                signal_type="bollinger",
                                signal_threshold=0.0
                            )



                        # STEP 1.2 预测 OHCLV

                        # 3. 开始遍历交易信号
                        # 根据日期范围筛选信号
                        if args.start_date:
                            start_dt = pd.to_datetime(args.start_date)
                            # 获取DataFrame的日期时区信息
                            tz = bollinger_signal_df['date'].dt.tz
                            # 将筛选日期转换为相同的时区
                            start_dt_tz = start_dt.tz_localize(tz)
                            bollinger_signal_df = bollinger_signal_df[bollinger_signal_df['date'] >= start_dt_tz]
                        if args.end_date:
                            end_dt = pd.to_datetime(args.end_date)
                            tz = bollinger_signal_df['date'].dt.tz
                            end_dt_tz = end_dt.tz_localize(tz)
                            bollinger_signal_df = bollinger_signal_df[bollinger_signal_df['date'] <= end_dt_tz]

                        all_metrics_list = []
                        all_results_df_list=[]
                        for _, trade_signal in bollinger_signal_df.iterrows():
                            trade_direction = trade_signal['signal']
                            signal_time = trade_signal['date']

                            # 4. 获取信号在1分钟数据中的索引
                            signal_idx = get_signal_index(raw_df_1min, signal_time)

                            logger.debug(f"信号索引 signal_idx = {signal_idx}, 数据总长度 = {len(raw_df_1min)}")



                            # 6. 执行所有窗口的回测
                            allowed_windows = interval_to_windows.get(time_interval, [1])
                            for execution_window in allowed_windows:
                                # 检查数据是否充足
                                if signal_idx + execution_window > len(raw_df_1min):
                                    print(
                                        f"[{selected_interval_label}] execution_window={execution_window} 数据不足，跳过。")
                                    continue

                                print(
                                    f"[{selected_interval_label}] 选定信号时间: {trade_signal['date']}，信号索引: {signal_idx}, 执行窗口: {execution_window}分钟"
                                )
                                print(
                                    f"[{selected_interval_label}] 交易方向: {trade_direction}, 总交易量: {total_quantity}, 失败安全系数: {failsafe_ratio}"
                                )
                                print(f"[{selected_interval_label}] boll参数: {bollinger_params}")
                                print(
                                    f"[{selected_interval_label}] 4. 执行回测... execution_window={execution_window}")

                                # 调用回测函数
                                results_df, metrics = run_adaptive_backtest(
                                    predictor_model,
                                    dataset_1min,
                                    raw_df_1min,
                                    signal_idx,
                                    model_input_seq_length,
                                    execution_window,
                                    total_quantity,
                                    trade_direction,
                                    failsafe_ratio,
                                    device,
                                    vae_model
                                )

                                logger.debug(f"回测结果 DataFrame 大小: {results_df.shape}")
                                logger.debug(f"回测指标 metrics: {metrics}")

                                # STEP 2： 计算回报与回撤
                                # 2.1 回报
                                # 从 metrics 字典中提取关键价格
                                adaptive_vwap_price = metrics.get("Adaptive VWAP Price", 0)
                                naive_vwap_price = metrics.get("naive VWAP", 0)

                                # 计算每单位的回报（与基准 VWAP 相比）
                                if trade_direction.lower() == "buy":
                                    # 买入策略：价格越低越好，所以用基准价减去实际成交价
                                    return_per_unit = naive_vwap_price - adaptive_vwap_price
                                else:  # "sell"
                                    # 卖出策略：价格越高越好，所以用实际成交价减去基准价
                                    return_per_unit = adaptive_vwap_price - naive_vwap_price

                                # 计算总回报
                                total_return = return_per_unit * total_quantity

                                print(f"该笔交易每单位回报为: {return_per_unit:.4f}")
                                print(f"该笔交易的总回报为: {total_return:.2f}")
                                metrics["Total Return"] = total_return

                                # 2.2 回撤
                                # 检查 results_df 是否为空，以避免错误
                                if results_df.empty:
                                    print("警告: 回测结果数据为空，无法计算回撤。")
                                    mae = 0.0
                                else:
                                    # 获取第一笔成交价作为进场基准价
                                    entry_price = results_df.iloc[0]['execution_price']
                                    trade_direction = metrics.get('trade_direction', 'buy')

                                    # 计算最大不利偏离 (MAE)
                                    if trade_direction.lower() == "buy":
                                        # 多头回撤：价格下跌
                                        min_price_during_trade = results_df['execution_price'].min()
                                        mae = (entry_price - min_price_during_trade) / entry_price
                                    else:  # "sell"
                                        # 空头回撤：价格上涨
                                        max_price_during_trade = results_df['execution_price'].max()
                                        mae = (max_price_during_trade - entry_price) / entry_price

                                print(f"该笔交易的最大回撤（MAE）为: {mae:.2%}")
                                metrics["Max Adverse Excursion (MAE)"] = mae

                                all_metrics_list.append(metrics)
                                all_results_df_list.append(results_df)

                        # --- 保存结果 ---
                        if all_metrics_list:
                            # 拼接所有结果 DataFrame
                            combined_results_df = pd.concat(all_results_df_list)

                            # 将指标列表转换为 DataFrame
                            combined_metrics_df = pd.DataFrame(all_metrics_list)

                            # 创建输出目录
                            output_dir = f"../../results/{trade_direction}_results"
                            os.makedirs(output_dir, exist_ok=True)

                            # 命名文件
                            results_df_name = f"results_df_{trade_direction}_{model_name}_{symbol}_{time_interval}_win{execution_window}.csv"
                            metrics_name = f"metrics_{trade_direction}_{model_name}_{symbol}_{time_interval}_win{execution_window}.csv"

                            # 保存合并后的 DataFrame
                            combined_results_df.to_csv(os.path.join(output_dir, results_df_name), index=True)
                            combined_metrics_df.to_csv(os.path.join(output_dir, metrics_name), index=True)

                            print(
                                f"[{selected_interval_label}] 所有回测完成，结果已保存为 {results_df_name} 和 {metrics_name}")




                    except Exception as e:
                        print(f"[{selected_interval_label}] 发生错误: {e}")
                        import traceback
                        print(traceback.format_exc())





