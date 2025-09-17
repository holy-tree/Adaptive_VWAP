import pandas as pd
import os
import pandas as pd
import os


# 定义时间筛选函数
def filter_by_time(df, start_date, end_date, time_col="timestamp"):
    """
    根据时间范围过滤 DataFrame
    参数:
        df: 输入的DataFrame
        start_date, end_date: datetime 类型
        time_col: 时间列的名字，默认是 'timestamp'
    返回:
        过滤后的 DataFrame
    """
    if time_col not in df.columns:
        print(f"警告: 找不到时间列 {time_col}，跳过筛选。")
        return df

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")  # 转换为时间格式
    mask = (df[time_col] >= start_date) & (df[time_col] <= end_date)
    return df.loc[mask].copy()


# 定义你的所有参数
base_path = '../results'  # 替换为你的实际路径，例如 'results'

# symbols = [
#     "rb9999", "i9999", "cu9999", "ni9999", "sc9999", "pg9999", "y9999", "ag9999", "m9999",
#     "c9999", "TA9999", "UR9999", "OI9999", "au9999", "IH9999", "T9999", "CF9999", "AP9999"
# ]
symbols = [
    # "rb9999","AP9999", "i9999","ni9999",
    "ni9999"
]

top_models = ["maa", "fits", "ptransformer","itransformer","macd"]
# top_models = ["fits"]
trades = [ "buy","sell"]
sub_models = ["rnn", "lstm", "our-transformer"]
# sub_models = ["our-transformer"]

intervals = [15, 30, 60]
execution_windows = [15, 30, 60]

start_date_str = '2025-04-01 00:00:00+08:00'
end_date_str = '2025-05-30 23:59:59+08:00'
start_date = pd.to_datetime(start_date_str)
end_date = pd.to_datetime(end_date_str)
# 遍历每个符号
for symbol in symbols:
    excel_filename = f'../results/summary/summary_table_{symbol}.xlsx'

    all_dfs = []

    # 遍历每对 interval 和 execution_window
    for interval in intervals:
        for execution_window in execution_windows:

            # 核心优化：确保执行窗口小于或等于周期
            if execution_window > interval:
                print(f"警告: 跳过不符合逻辑的组合: interval={interval}, execution_window={execution_window}")
                continue

            result_df = pd.DataFrame(index=top_models)

            for trade in trades:
                for sub_model in sub_models:
                    mean_values = []
                    var_values = []
                    # 为 Adaptive VWAP Price 准备列表
                    adaptive_vwap_means = []

                    # 为 naive VWAP 准备列表
                    naive_vwap_means = []

                    total_return_means = []
                    mae_means = []

                    for top_model in top_models:
                        path = os.path.join(
                            base_path,
                            top_model,
                            trade,
                            sub_model,
                            symbol,
                            f'{interval}min_win{execution_window}',
                            'metrics.csv'
                        )

                        try:
                            metrics_df = pd.read_csv(path)
                            metrics_df = filter_by_time(metrics_df, start_date, end_date, time_col="start_time")

                            slippage_data = metrics_df['Slippage Reduction (BPS)']
                            mean_val = slippage_data.mean()
                            var_val = slippage_data.var()
                            mean_values.append(mean_val)
                            var_values.append(var_val)

                            # 计算 Adaptive VWAP Price 的均值
                            adaptive_vwap_means.append(metrics_df['Adaptive VWAP Price'].mean())

                            # 计算 naive VWAP 的均值
                            naive_vwap_means.append(metrics_df['naive VWAP'].mean())

                            total_return_means.append(metrics_df['Total Return'].mean())
                            mae_means.append(metrics_df['Max Adverse Excursion (MAE)'].mean())

                        except FileNotFoundError:
                            print(f"警告: 路径不存在 - {path}")
                            mean_values.append(None)
                            var_values.append(None)
                            adaptive_vwap_means.append(None)
                            naive_vwap_means.append(None)
                            total_return_means.append(None)
                            mae_means.append(None)
                        except KeyError:
                            print(f"警告: 文件中缺少'Slippage Reduction (BPS)'列 - {path}")
                            mean_values.append(None)
                            var_values.append(None)
                            adaptive_vwap_means.append(None)
                            naive_vwap_means.append(None)
                            total_return_means.append(None)
                            mae_means.append(None)

                    column_name_mean = f'{trade}-{sub_model}_BPS_mean'
                    # column_name_var = f'{trade}-{sub_model}_variance'
                    result_df[column_name_mean] = mean_values
                    # result_df[column_name_var] = var_values

                    # 添加 Adaptive VWAP Price 列
                    adaptive_vwap_col_name = f'{trade}-{sub_model}_VWAP_mean'
                    result_df[adaptive_vwap_col_name] = adaptive_vwap_means

                    # 添加 naive VWAP 列
                    naive_vwap_col_name = f'{trade}-{sub_model}_Naive_VWAP_mean'
                    result_df[naive_vwap_col_name] = naive_vwap_means

                    result_df[f'{trade}-{sub_model}_TotalReturn_mean'] = total_return_means
                    result_df[f'{trade}-{sub_model}_MAE_mean'] = mae_means

            # 为每个表格添加一个标题行
            title_df = pd.DataFrame(
                [[f'interval-{interval}_execution_windows-{execution_window}']],
                columns=['Summary']
            )

            # 为每个表格添加一个空行，方便区分
            empty_row_df = pd.DataFrame([['']], columns=[''])

            # 将标题、结果表格和空行按顺序添加到列表中
            all_dfs.extend([title_df, result_df, empty_row_df])
            print(f"已为 {symbol} 生成表格: I{interval}_W{execution_window}")

    # 将所有 DataFrame 合并为一个大的 DataFrame
    # 这一步将所有的表格按顺序拼接，确保了结构完整性
    full_df = pd.concat(all_dfs, ignore_index=False)

    # 将完整的 DataFrame 写入 Excel
    full_df.to_excel(excel_filename, index=True, sheet_name='Summary')

    print(f"所有关于 {symbol} 的表格已保存在 {excel_filename} 中。")