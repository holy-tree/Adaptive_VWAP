# import pandas as pd
# import os
#
# # 定义你的所有参数
# base_path = '../results'  # 替换为你的实际路径，例如 'results'
#
# symbols = [
#     "rb9999", "i9999", "cu9999", "ni9999", "sc9999", "pg9999", "y9999", "ag9999", "m9999",
#     "c9999", "TA9999", "UR9999", "OI9999", "au9999", "IH9999", "T9999", "CF9999", "AP9999"
# ]
#
# top_models = ["maa", "itransformer", "fits", "ptransformer"]
# trades = ["buy", "sell"]
# sub_models = ["lstm", "rnn", "our-transformer"]
#
# intervals = [1, 5, 15, 30, 60]
# execution_windows = [1, 5, 15, 30, 60]
#
# # 遍历每个符号
# for symbol in symbols:
#     # 为每个符号创建一个Excel文件，以便存储所有表格
#     excel_filename = f'../results/summary/summary_table_{symbol}.xlsx'
#
#     # 使用 Pandas 的 ExcelWriter
#     with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
#
#         # 遍历每对 interval 和 execution_window
#         for interval in intervals:
#             for execution_window in execution_windows:
#
#                 # 创建一个空的 DataFrame 来存放当前组合的结果
#                 result_df = pd.DataFrame(index=top_models)
#
#                 # 遍历每种交易类型 (buy/sell)
#                 for trade in trades:
#                     # 遍历每种底层模型 (lstm/rnn/our-transformer)
#                     for sub_model in sub_models:
#
#                         # 遍历每种顶层模型 (maa, itransformer, ...)
#                         mean_values = []
#                         var_values = []
#
#                         for top_model in top_models:
#                             # 构建完整的文件路径
#                             path = os.path.join(
#                                 base_path,
#                                 top_model,
#                                 trade,
#                                 sub_model,
#                                 symbol,
#                                 f'{interval}min_win{execution_window}',
#                                 'metrics.csv'
#                             )
#
#                             try:
#                                 # 读取 metrics.csv 文件
#                                 metrics_df = pd.read_csv(path)
#
#                                 # 提取 Slippage Reduction (BPS) 列
#                                 slippage_data = metrics_df['Slippage Reduction (BPS)']
#
#                                 # 计算均值和方差
#                                 mean_val = slippage_data.mean()
#                                 var_val = slippage_data.var()
#
#                                 mean_values.append(mean_val)
#                                 var_values.append(var_val)
#
#                             except FileNotFoundError:
#                                 print(f"警告: 路径不存在 - {path}")
#                                 mean_values.append(None)
#                                 var_values.append(None)
#                             except KeyError:
#                                 print(f"警告: 文件中缺少'Slippage Reduction (BPS)'列 - {path}")
#                                 mean_values.append(None)
#                                 var_values.append(None)
#
#                         # 将计算出的均值和方差添加到结果 DataFrame
#                         column_name_mean = f'{trade}-{sub_model}_mean'
#                         column_name_var = f'{trade}-{sub_model}_variance'
#
#                         result_df[column_name_mean] = mean_values
#                         result_df[column_name_var] = var_values
#
#                 # 将当前组合的表格写入 Excel 的一个新 sheet
#                 sheet_name = f'I{interval}_W{execution_window}'
#                 result_df.to_excel(writer, sheet_name=sheet_name)
#                 print(f"已为 {symbol} 生成表格: {sheet_name}")
#
#     print(f"所有关于 {symbol} 的表格已保存在 {excel_filename} 中。")

import pandas as pd
import os

# 定义你的所有参数
base_path = '../results'  # 替换为你的实际路径，例如 'results'

symbols = [
    "rb9999", "i9999", "cu9999", "ni9999", "sc9999", "pg9999", "y9999", "ag9999", "m9999",
    "c9999", "TA9999", "UR9999", "OI9999", "au9999", "IH9999", "T9999", "CF9999", "AP9999"
]

# top_models = ["maa", "itransformer", "fits", "ptransformer"]
top_models = ["maa", "itransformer",  "ptransformer"]
trades = ["buy", "sell"]
sub_models = ["lstm", "rnn", "our-transformer"]

intervals = [1, 5, 15, 30, 60]
execution_windows = [1, 5, 15, 30, 60]

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
                            slippage_data = metrics_df['Slippage Reduction (BPS)']
                            mean_val = slippage_data.mean()
                            var_val = slippage_data.var()
                            mean_values.append(mean_val)
                            var_values.append(var_val)
                        except FileNotFoundError:
                            print(f"警告: 路径不存在 - {path}")
                            mean_values.append(None)
                            var_values.append(None)
                        except KeyError:
                            print(f"警告: 文件中缺少'Slippage Reduction (BPS)'列 - {path}")
                            mean_values.append(None)
                            var_values.append(None)

                    column_name_mean = f'{trade}-{sub_model}_mean'
                    column_name_var = f'{trade}-{sub_model}_variance'
                    result_df[column_name_mean] = mean_values
                    result_df[column_name_var] = var_values

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