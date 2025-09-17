import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

interval_to_windows = {
    # '1min': [1],
    # '5min': [1, 5],
    '15min': [15],
    '30min': [15,30],
    '60min': [15, 30, 60],
}

time_interval_map = {
    '1分钟': '1min', '5分钟': '5min', '15分钟': '15min',
    '30分钟': '30min', '1小时': '60min', '日线': 'D'
}

base_path = "../all_results"
symbols = [
    "rb9999", "i9999", "cu9999", "ni9999", "sc9999",
    "pg9999", "y9999", "ag9999", "m9999", "c9999",
    "TA9999", "UR9999", "OI9999", "au9999", "IH9999",
    "T9999", "CF9999", "AP9999"
]
top_models = ["maa", "itransformer", "fits", "ptransformer", "macd"]
trades = ["buy", "sell"]
sub_models = ["our-transformer"]

periods = ["60min"]
periods_windows = {"5min": [4,],
            "15min": [10, 12],
            "30min": [20, 25],
            '60min': [30, 50],
            "D": [30, 60, 150]}


def build_table_data33(direction, symbol, period_symbol_metrics_pathes, period_symbol_df_pathes, period_symbol_pathes):
    """
    对每个周期period，分别根据 signals/strategy/direction/symbol_period_xxx.csv 文件，计算每个策略的 return 和 retracement。
    每个周期分别保存为 {symbol}_{direction}_{period}_signals_table.csv
    并综合计算所有模型/策略/周期的表格，保存为 {symbol}_{direction}_all_signals_table.csv

    # 所有资产的价格都用 execution_price = (high+low+close)/3.0
    """
    strategy_map = {
        "macd": "MACD",
        "itransformer": "iTransformer",
        "fits": "FITS",
        "ptransformer": "PatchTST",
        "maa": "MAA"
    }
    table_strategies = ["maa", "itransformer", "fits", "ptransformer", "macd"]
    periods_order = ["5min", "15min", "60min", "D"]
    period_label = {"5min": "5min", "15min": "15min", "60min": "60min", "D": "1day"}
    model_types = [
        ("naïve VWAP", "naive"),
        ("Ours- lstm", "lstm"),
        ("Ours- rnn", "rnn"),
        ("Ours- Transformer", "transformer"),
    ]
    all_results = {period: {model[0]: {metric: {strat: None for strat in table_strategies} for metric in ["Slippage", "Return", "Retracement"]} for model in model_types} for period in periods_order}

    for period in periods_order:
        table_data = []
        table_data.append(["Signal Freq", "", *table_strategies])
        table_data.append(["Strategy", "", *[""]*6])
        for metric in ["Return", "Retracement"]:
            row = ["", metric] + [""]*6
            table_data.append(row)
        table_data[-3][0] = "Signal Freq"
        table_data[-3][1] = period_label[period]

        for sidx, strat in enumerate(table_strategies):
            strat_key = [k for k, v in strategy_map.items() if v == strat]
            if not strat_key:
                continue
            strat_key = strat_key[0]
            strat_dir = f"./signals/{strat_key}/{direction}"
            pattern = f"{strat_dir}/{symbol}_{strat_key}_signals_{period}_{direction}.csv"
            files = glob.glob(pattern)
            if not files:
                if strat_key == "GAN":
                    pattern = f"{strat_dir}/{symbol}_GAN_signals_{period}_{direction}.csv"
                    files = glob.glob(pattern)
            if not files:
                continue
            file = files[0]
            try:
                df = pd.read_csv(file)
            except Exception:
                continue
            if df.empty or "date" not in df.columns or "signal" not in df.columns:
                continue
            df = df.sort_values("date").reset_index(drop=True)
            signals = df[["date", "signal"]].copy()
            position = 0
            entry_price = None
            prices = []
            # 计算 execution_price
            if all(col in df.columns for col in ["high", "low", "close"]):
                price_arr = (df["high"] + df["low"] + df["close"]) / 3.0
                price_arr = price_arr.values
            else:
                continue
            for i, row in signals.iterrows():
                sig = row["signal"]
                price = price_arr[i]
                if sig in ["Buy"]:
                    if position == 0:
                        entry_price = price
                        position = 1
                        prices.append(entry_price)
                elif sig in ["Sell"]:
                    if position == 0:
                        entry_price = price
                        position = -1
                        prices.append(entry_price)
                elif sig in ["Close_Buy"]:
                    if position == 1 and entry_price is not None:
                        prices.append(price)
                        position = 0
                        entry_price = None
                elif sig in ["Close_Sell"]:
                    if position == -1 and entry_price is not None:
                        prices.append(price)
                        position = 0
                        entry_price = None
            if position != 0 and entry_price is not None:
                prices.append(price_arr[-1])
            returns = []
            for i in range(1, len(prices), 2):
                p0 = prices[i-1]
                p1 = prices[i]
                if direction == "buy":
                    ret = (p1 - p0) / p0 if p0 != 0 else 0
                else:
                    ret = (p0 - p1) / p0 if p0 != 0 else 0
                returns.append(ret)
            total_return = np.sum(returns) if returns else 0
            curve = [1]
            for r in returns:
                curve.append(curve[-1] * (1 + r))
            mdd = 0
            peak = curve[0] if curve else 1
            for val in curve:
                if val > peak:
                    peak = val
                dd = (peak - val) / peak if peak != 0 else 0
                if dd > mdd:
                    mdd = dd
            table_data[-2][2 + sidx] = f"{total_return:.4f}"
            table_data[-1][2 + sidx] = f"{mdd:.4f}"

        df_out = pd.DataFrame(table_data)
        out_path = f"{symbol}_{direction}_{period}_signals_table.csv"
        df_out.to_csv(out_path, index=False, header=False, encoding="utf-8-sig")
        print(f"Saved signals table for {symbol} {period} to {out_path}")

    # --------- 修正naive和transformer数据相同的问题 -----------
    # 原因分析：
    # 你在下面的metrics读取部分，metrics_file的选择是通过model_key in f.lower()，但naive和transformer的metrics文件通常是同一个文件（或者transformer的metrics文件名里也包含naive），
    # 或者你根本没有单独的naive metrics文件，导致两者都用transformer的metrics。
    # 另外，naive VWAP的slippage应该是直接用Benchmark VWAP，不应该用模型的slippage。
    # 修正方法：
    # - naive VWAP的slippage直接设为0或nan（因为它是基准，不存在slippage）。
    # - naive VWAP的return和retracement应单独用Benchmark VWAP做信号回测，不应用模型信号。
    # - 只有模型（transformer/lstm/rnn）才用metrics文件的slippage。
    # - 如果metrics文件名区分不清楚，建议严格区分文件名或在此处逻辑区分。
    # 下面做如下修正：

    for period in periods_order:
        for model_name, model_key in model_types:
            for strat in table_strategies:
                strat_key = [k for k, v in strategy_map.items() if v == strat]
                if not strat_key:
                    continue
                strat_key = strat_key[0]
                slip = None
                ret = None
                mdd = None
                if model_key == "naive":
                    # naive VWAP: slippage为0，return和retracement用Benchmark VWAP信号回测
                    slip = 0
                    # 找到信号文件
                    strat_dir = f"./signals/{strat_key}/{direction}"
                    pattern = f"{strat_dir}/{symbol}_{strat_key}_signals_{period}_{direction}.csv"
                    files2 = glob.glob(pattern)
                    if not files2 and strat_key == "GAN":
                        pattern = f"{strat_dir}/{symbol}_GAN_signals_{period}_{direction}.csv"
                        files2 = glob.glob(pattern)
                    if files2:
                        try:
                            df = pd.read_csv(files2[0])
                            if df.empty or not all(col in df.columns for col in ["high", "low", "close"]):
                                raise Exception()
                            # 用Benchmark VWAP做回测（假设有该列，否则用HLC/3）
                            if "Benchmark VWAP" in df.columns:
                                price_arr = df["Benchmark VWAP"].values
                            else:
                                price_arr = ((df["high"] + df["low"] + df["close"]) / 3.0).values
                            signals = df["signal"].values
                            position = 0
                            entry_price = None
                            prices = []
                            for i, sig in enumerate(signals):
                                price = price_arr[i]
                                if sig == "Buy":
                                    if position == 0:
                                        entry_price = price
                                        position = 1
                                        prices.append(entry_price)
                                elif sig == "Sell":
                                    if position == 0:
                                        entry_price = price
                                        position = -1
                                        prices.append(entry_price)
                                elif sig == "Close_Buy":
                                    if position == 1 and entry_price is not None:
                                        prices.append(price)
                                        position = 0
                                        entry_price = None
                                elif sig == "Close_Sell":
                                    if position == -1 and entry_price is not None:
                                        prices.append(price)
                                        position = 0
                                        entry_price = None
                            if position != 0 and entry_price is not None:
                                prices.append(price_arr[-1])
                            returns = []
                            for i in range(1, len(prices), 2):
                                p0 = prices[i-1]
                                p1 = prices[i]
                                if direction == "buy":
                                    r = (p1 - p0) / p0 if p0 != 0 else 0
                                else:
                                    r = (p0 - p1) / p0 if p0 != 0 else 0
                                returns.append(r)
                            ret = np.sum(returns) if returns else 0
                            curve = [1]
                            for r in returns:
                                curve.append(curve[-1] * (1 + r))
                            mdd_val = 0
                            peak = curve[0] if curve else 1
                            for val in curve:
                                if val > peak:
                                    peak = val
                                dd = (peak - val) / peak if peak != 0 else 0
                                if dd > mdd_val:
                                    mdd_val = dd
                            mdd = mdd_val
                        except Exception:
                            ret = None
                            mdd = None
                else:
                    # 模型slippage
                    files = period_symbol_metrics_pathes.get(period, {}).get(symbol, [])
                    metrics_file = None
                    for f in files:
                        # 只匹配当前模型
                        if strat_key in f and model_key in f.lower():
                            metrics_file = f
                            break
                    if metrics_file:
                        try:
                            metrics = pd.read_csv(metrics_file)
                            if "Slippage Reduction (BPS)" in metrics.columns:
                                slip = metrics["Slippage Reduction (BPS)"].mean()
                        except Exception:
                            slip = None
                    # return和retracement还是用信号文件
                    strat_dir = f"./signals/{strat_key}/{direction}"
                    pattern = f"{strat_dir}/{symbol}_{strat_key}_signals_{period}_{direction}.csv"
                    files2 = glob.glob(pattern)
                    if not files2 and strat_key == "GAN":
                        pattern = f"{strat_dir}/{symbol}_GAN_signals_{period}_{direction}.csv"
                        files2 = glob.glob(pattern)
                    if files2:
                        try:
                            df = pd.read_csv(files2[0])
                            if df.empty or not all(col in df.columns for col in ["high", "low", "close"]):
                                raise Exception()
                            price_arr = ((df["high"] + df["low"] + df["close"]) / 3.0).values
                            signals = df["signal"].values
                            position = 0
                            entry_price = None
                            prices = []
                            for i, sig in enumerate(signals):
                                price = price_arr[i]
                                if sig == "Buy":
                                    if position == 0:
                                        entry_price = price
                                        position = 1
                                        prices.append(entry_price)
                                elif sig == "Sell":
                                    if position == 0:
                                        entry_price = price
                                        position = -1
                                        prices.append(entry_price)
                                elif sig == "Close_Buy":
                                    if position == 1 and entry_price is not None:
                                        prices.append(price)
                                        position = 0
                                        entry_price = None
                                elif sig == "Close_Sell":
                                    if position == -1 and entry_price is not None:
                                        prices.append(price)
                                        position = 0
                                        entry_price = None
                            if position != 0 and entry_price is not None:
                                prices.append(price_arr[-1])
                            returns = []
                            for i in range(1, len(prices), 2):
                                p0 = prices[i-1]
                                p1 = prices[i]
                                if direction == "buy":
                                    r = (p1 - p0) / p0 if p0 != 0 else 0
                                else:
                                    r = (p0 - p1) / p0 if p0 != 0 else 0
                                returns.append(r)
                            ret = np.sum(returns) if returns else 0
                            curve = [1]
                            for r in returns:
                                curve.append(curve[-1] * (1 + r))
                            mdd_val = 0
                            peak = curve[0] if curve else 1
                            for val in curve:
                                if val > peak:
                                    peak = val
                                dd = (peak - val) / peak if peak != 0 else 0
                                if dd > mdd_val:
                                    mdd_val = dd
                            mdd = mdd_val
                        except Exception:
                            ret = None
                            mdd = None
                all_results[period][model_name]["Slippage"][strat] = slip
                all_results[period][model_name]["Return"][strat] = ret
                all_results[period][model_name]["Retracement"][strat] = mdd

    all_table = []
    for period in periods_order:
        all_table.append(["Signal Freq", "", *table_strategies])
        all_table.append(["Signal Freq", period_label[period], *[""]*6])
        for model_name, _ in model_types:
            for metric in ["Slippage", "Return", "Retracement"]:
                row = [model_name, metric]
                for strat in table_strategies:
                    val = all_results[period][model_name][metric][strat]
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        row.append("")
                    else:
                        row.append(f"{val:.4f}")
                all_table.append(row)
    df_all = pd.DataFrame(all_table)
    out_path_all = f"{symbol}_{direction}_all_signals_table.csv"
    df_all.to_csv(out_path_all, index=False, header=False, encoding="utf-8-sig")
    print(f"Saved all signals summary table for {symbol} to {out_path_all}")
    return None

def plot_all_metrics_curves(symbol, periods, period_symbol_metrics_pathes, direction, period_symbol_df_pathes=None, base_path=None):
    """
    绘制所有品种、周期、策略的指标曲线。

    【Slippage计算方式说明】:
    - 当前slippage（滑点）的计算方式为：买入窗口内各个hcl价格减去总signal单位的hcl比上总的signal单位的hcl。
      即：slippage = (sum(HLC/3 * signal_units) / sum(signal_units) - sum(HLC/3 * signal_units) / sum(signal_units)) / sum(HLC/3 * signal_units) / sum(signal_units)
      这里 signal_units 指每个信号的单位数量，假设为1，则为窗口内HLC/3均值与信号均值的相对差异。
    - 其它指标和逻辑不变。
    """

    import plotly.graph_objects as go
    import plotly.express as px

    strategy_map = {
        "macd": "MACD",
        "itransformer": "iTransformer",
        "fits": "FITS",
        "ptransformer": "PatchTST",
        "maa": "MAA"
    }
    table_strategies = ["MACD", "BOLL", "RSI", "Dual Thrust", "KDJ", "MAA"]
    model_types = ["RNN", "LSTM", "Transformer"]
    metric_names = ["Slippage", "Return", "Retracement"]

    color_map = {
        "MACD": "#636EFA", "BOLL": "#EF553B", "RSI": "#00CC96",
        "Dual Thrust": "#AB63FA", "KDJ": "#FFA15A", "MAA": "#19D3F3"
    }
    model_dash = {"RNN": "dot", "LSTM": "dash", "Transformer": "solid"}
    model_color = {"RNN": "#636EFA", "LSTM": "#EF553B", "Transformer": "#00CC96"}

    print("Start plotting all metrics curves...")
    for period in tqdm(periods, desc=f"Periods ({symbol})", leave=False):
            print(f"Processing: Symbol={symbol}, Period={period}, Direction={direction}")
            metric_curves_index = {metric: go.Figure() for metric in metric_names}
            metric_curves_time = {metric: go.Figure() for metric in metric_names}
            slippage_dist, return_dist, retracement_dist = [], [], []
            kline_signal_data = []

            # 新增指标数据收集
            signal_freq_list = []
            win_rate_list = []
            avg_holding_period_list = []
            profit_factor_list = []

            for strat in tqdm(table_strategies, desc=f"Strategies ({symbol}-{period})", leave=False):
                strat_key = [k for k, v in strategy_map.items() if v == strat]
                if not strat_key:
                    continue
                strat_key = strat_key[0]
                strat_dir = f"./signals/{strat_key}/{direction}"
                pattern = f"{strat_dir}/{symbol}_{strat_key}_signals_{period}_{direction}.csv"
                files2 = glob.glob(pattern)
                if not files2 and strat_key == "GAN":
                    pattern = f"{strat_dir}/{symbol}_GAN_signals_{period}_{direction}.csv"
                    files2 = glob.glob(pattern)
                if not files2:
                    continue
                try:
                    df = pd.read_csv(files2[0])
                except Exception:
                    continue
                if df.empty or not all(col in df.columns for col in ["high", "low", "close", "signal"]):
                    continue
                price_arr = (df["high"] + df["low"] + df["close"]) / 3.0
                signals = df["signal"].values
                position = 0
                entry_price = None
                prices = []
                entry_idx = None
                holding_periods = []
                trade_results = []
                kline_signal_data.append((strat, df.copy()))
                slippage_list = []

                # 新slippage定义：买入窗口内各个hcl价格减去总signal单位的hcl比上总的signal单位的hcl
                # 这里假设每个信号单位为1，窗口为每次持仓区间
                window_start = None
                for i, sig in enumerate(signals):
                    price = price_arr.iloc[i]
                    if sig == "Buy":
                        if position == 0:
                            entry_price = price
                            entry_idx = i
                            position = 1
                            prices.append(entry_price)
                            window_start = i
                    elif sig == "Sell":
                        if position == 0:
                            entry_price = price
                            entry_idx = i
                            position = -1
                            prices.append(entry_price)
                            window_start = i
                    elif sig == "Close_Buy":
                        if position == 1 and entry_price is not None:
                            prices.append(price)
                            # 新slippage计算
                            if window_start is not None:
                                window_prices = price_arr.iloc[window_start:i+1]
                                signal_units = np.ones(len(window_prices))
                                avg_hcl = np.sum(window_prices * signal_units) / np.sum(signal_units)
                                # 理论成交价为窗口内均值
                                slippage = (avg_hcl - entry_price) / entry_price * 10000 if entry_price != 0 else 0
                                slippage_list.append(slippage)
                            # 新增：记录持仓周期和盈亏
                            if entry_idx is not None:
                                holding_periods.append(i - entry_idx)
                            trade_results.append(price - entry_price)
                            position = 0
                            entry_price = None
                            entry_idx = None
                            window_start = None
                    elif sig == "Close_Sell":
                        if position == -1 and entry_price is not None:
                            prices.append(price)
                            if window_start is not None:
                                window_prices = price_arr.iloc[window_start:i+1]
                                signal_units = np.ones(len(window_prices))
                                avg_hcl = np.sum(window_prices * signal_units) / np.sum(signal_units)
                                slippage = (entry_price - avg_hcl) / entry_price * 10000 if entry_price != 0 else 0
                                slippage_list.append(slippage)
                            if entry_idx is not None:
                                holding_periods.append(i - entry_idx)
                            trade_results.append(entry_price - price)
                            position = 0
                            entry_price = None
                            entry_idx = None
                            window_start = None
                # If still holding position at the end, close at last price
                if position != 0 and entry_price is not None:
                    last_price = price_arr.iloc[-1]
                    prices.append(last_price)
                    if window_start is not None:
                        window_prices = price_arr.iloc[window_start:len(signals)]
                        signal_units = np.ones(len(window_prices))
                        avg_hcl = np.sum(window_prices * signal_units) / np.sum(signal_units)
                        if position == 1:
                            slippage = (avg_hcl - entry_price) / entry_price * 10000 if entry_price != 0 else 0
                            slippage_list.append(slippage)
                        elif position == -1:
                            slippage = (entry_price - avg_hcl) / entry_price * 10000 if entry_price != 0 else 0
                            slippage_list.append(slippage)
                    if position == 1:
                        if entry_idx is not None:
                            holding_periods.append(len(signals) - 1 - entry_idx)
                        trade_results.append(last_price - entry_price)
                    elif position == -1:
                        if entry_idx is not None:
                            holding_periods.append(len(signals) - 1 - entry_idx)
                        trade_results.append(entry_price - last_price)
                for v in slippage_list:
                    slippage_dist.append({"Strategy": strat, "Model": "Signal", "Value": v})

                # 新增指标统计
                # 1. Signal Frequency
                signal_freq = np.sum([sig in ["Buy", "Sell", "Close_Buy", "Close_Sell"] for sig in signals])
                signal_freq_list.append({"Strategy": strat, "Value": signal_freq})
                # 2. Win Rate
                win_trades = [r for r in trade_results if r > 0]
                total_trades = len(trade_results)
                win_rate = len(win_trades) / total_trades if total_trades > 0 else np.nan
                win_rate_list.append({"Strategy": strat, "Value": win_rate * 100 if win_rate is not np.nan else np.nan})
                # 3. Average Holding Period
                avg_holding = np.mean(holding_periods) if holding_periods else np.nan
                avg_holding_period_list.append({"Strategy": strat, "Value": avg_holding})
                # 4. Profit Factor
                gross_profit = np.sum([r for r in trade_results if r > 0])
                gross_loss = np.abs(np.sum([r for r in trade_results if r < 0]))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
                profit_factor_list.append({"Strategy": strat, "Value": profit_factor})

                # --- Return and Retracement calculation (unchanged) ---
                returns = []
                for i in range(1, len(prices), 2):
                    p0 = prices[i-1]
                    p1 = prices[i]
                    if direction == "buy":
                        ret = (p1 - p0) / p0 if p0 != 0 else 0
                    else:
                        ret = (p0 - p1) / p0 if p0 != 0 else 0
                    returns.append(ret)
                curve = [1]
                for r in returns:
                    curve.append(curve[-1] * (1 + r))
                drawdown_curve = []
                peak = curve[0]
                for val in curve:
                    if val > peak:
                        peak = val
                    dd = (peak - val) / peak if peak != 0 else 0
                    drawdown_curve.append(dd)
                line_width = 1
                line_dash = "solid"
                line_color = color_map.get(strat, None)
                metric_curves_index["Return"].add_trace(go.Scatter(
                    x=list(range(len(curve))), y=curve, mode='lines', name=f"{strat}",
                    line=dict(width=line_width, color=line_color, dash=line_dash)
                ))
                metric_curves_index["Retracement"].add_trace(go.Scatter(
                    x=list(range(len(drawdown_curve))), y=drawdown_curve, mode='lines', name=f"{strat}",
                    line=dict(width=line_width, color=line_color, dash=line_dash)
                ))
                if "date" in df.columns:
                    time_x = pd.to_datetime(df["date"])
                    time_x_curve = time_x.iloc[:len(curve)]
                    time_x_drawdown = time_x.iloc[:len(drawdown_curve)]
                    metric_curves_time["Return"].add_trace(go.Scatter(
                        x=time_x_curve, y=curve, mode='lines', name=f"{strat}",
                        line=dict(width=line_width, color=line_color, dash=line_dash)
                    ))
                    metric_curves_time["Retracement"].add_trace(go.Scatter(
                        x=time_x_drawdown, y=drawdown_curve, mode='lines', name=f"{strat}",
                        line=dict(width=line_width, color=line_color, dash=line_dash)
                    ))
                return_dist.extend([{"Strategy": strat, "Value": v} for v in returns])
                retracement_dist.extend([{"Strategy": strat, "Value": v} for v in drawdown_curve])

            # 原有指标曲线绘制（不变）
            for metric in metric_names:
                fig = metric_curves_index[metric]
                fig.update_layout(
                    title=f"{symbol} {period} {direction} {metric} (Index-X)",
                    xaxis_title="Index",
                    yaxis_title=metric,
                    legend_title="Strategy",
                    width=900, height=400,
                    template="simple_white", font=dict(size=13, family="Arial"),
                    margin=dict(l=40, r=10, t=40, b=40), legend=dict(font=dict(size=11))
                )
                out_path = f"./images/{symbol}_{period}_{direction}_{metric.lower()}_curve_index.png"
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                if len(fig.data) > 0:
                    fig.write_image(out_path, scale=2)
                fig_time = metric_curves_time[metric]
                fig_time.update_layout(
                    title=f"{symbol} {period} {direction} {metric} (Time-X)",
                    xaxis_title="Time",
                    yaxis_title=metric,
                    legend_title="Strategy",
                    width=900, height=400,
                    template="simple_white", font=dict(size=13, family="Arial"),
                    margin=dict(l=40, r=10, t=40, b=40), legend=dict(font=dict(size=11))
                )
                out_path_time = f"./images/{symbol}_{period}_{direction}_{metric.lower()}_curve_time.png"
                if len(fig_time.data) > 0:
                    fig_time.write_image(out_path_time, scale=2)

            # 新增指标图表
            if signal_freq_list:
                df_freq = pd.DataFrame(signal_freq_list)
                fig = px.bar(df_freq, x="Strategy", y="Value", color="Strategy",
                             title=f"{symbol} {period} {direction} Signal Frequency",
                             color_discrete_map=color_map)
                fig.update_layout(width=700, height=350, template="simple_white", font=dict(size=12),
                                 xaxis_title="Strategy", yaxis_title="Signal Count", legend_title="Strategy")
                freq_path = f"./images/{symbol}_{period}_{direction}_signal_frequency.png"
                fig.write_image(freq_path, scale=2)
            if win_rate_list:
                df_win = pd.DataFrame(win_rate_list)
                fig = px.bar(df_win, x="Strategy", y="Value", color="Strategy",
                             title=f"{symbol} {period} {direction} Win Rate",
                             color_discrete_map=color_map)
                fig.update_layout(width=700, height=350, template="simple_white", font=dict(size=12),
                                 xaxis_title="Strategy", yaxis_title="Win Rate (%)", legend_title="Strategy",
                                 yaxis=dict(range=[0, 100]))
                win_path = f"./images/{symbol}_{period}_{direction}_win_rate.png"
                fig.write_image(win_path, scale=2)
            if avg_holding_period_list:
                df_hold = pd.DataFrame(avg_holding_period_list)
                fig = px.bar(df_hold, x="Strategy", y="Value", color="Strategy",
                             title=f"{symbol} {period} {direction} Avg Holding Period",
                             color_discrete_map=color_map)
                fig.update_layout(width=700, height=350, template="simple_white", font=dict(size=12),
                                 xaxis_title="Strategy", yaxis_title="Avg Holding Period (bars)", legend_title="Strategy")
                hold_path = f"./images/{symbol}_{period}_{direction}_avg_holding_period.png"
                fig.write_image(hold_path, scale=2)
            if profit_factor_list:
                df_pf = pd.DataFrame(profit_factor_list)
                fig = px.bar(df_pf, x="Strategy", y="Value", color="Strategy",
                             title=f"{symbol} {period} {direction} Profit Factor",
                             color_discrete_map=color_map)
                fig.update_layout(width=700, height=350, template="simple_white", font=dict(size=12),
                                 xaxis_title="Strategy", yaxis_title="Profit Factor", legend_title="Strategy")
                pf_path = f"./images/{symbol}_{period}_{direction}_profit_factor.png"
                fig.write_image(pf_path, scale=2)

            # 原有分布类图表（不变）
            if slippage_dist:
                df_slip = pd.DataFrame(slippage_dist)
                fig_box = px.box(df_slip, x="Strategy", y="Value", color="Model", points="all",
                                    title=f"{symbol} {period} {direction} Slippage Boxplot",
                                    color_discrete_map=model_color)
                fig_box.update_layout(width=700, height=350, template="simple_white", font=dict(size=12),
                                     xaxis_title="Strategy", yaxis_title="Slippage (BPS)", legend_title="Model")
                box_path = f"./images/{symbol}_{period}_{direction}_slippage_boxplot.png"
                fig_box.write_image(box_path, scale=2)
                fig_violin = px.violin(df_slip, x="Strategy", y="Value", color="Model", box=True, points="all",
                                        title=f"{symbol} {period} {direction} Slippage Violin",
                                        color_discrete_map=model_color)
                fig_violin.update_layout(width=700, height=350, template="simple_white", font=dict(size=12),
                                        xaxis_title="Strategy", yaxis_title="Slippage (BPS)", legend_title="Model")
                violin_path = f"./images/{symbol}_{period}_{direction}_slippage_violin.png"
                fig_violin.write_image(violin_path, scale=2)
                fig_density = px.density_contour(df_slip, x="Value", color="Strategy",
                                                    title=f"{symbol} {period} {direction} Slippage Density")
                fig_density.update_layout(width=700, height=350, template="simple_white", font=dict(size=12),
                                         xaxis_title="Slippage (BPS)", legend_title="Strategy")
                density_path = f"./images/{symbol}_{period}_{direction}_slippage_density.png"
                fig_density.write_image(density_path, scale=2)
                fig_ecdf = px.ecdf(df_slip, x="Value", color="Strategy",
                                    title=f"{symbol} {period} {direction} Slippage ECDF")
                fig_ecdf.update_layout(width=700, height=350, template="simple_white", font=dict(size=12),
                                      xaxis_title="Slippage (BPS)", legend_title="Strategy")
                ecdf_path = f"./images/{symbol}_{period}_{direction}_slippage_ecdf.png"
                fig_ecdf.write_image(ecdf_path, scale=2)
                fig_hist = px.histogram(df_slip, x="Value", color="Strategy", barmode="overlay",
                                        nbins=50, marginal="violin",
                                        title=f"{symbol} {period} {direction} Slippage Histogram")
                fig_hist.update_layout(width=700, height=350, template="simple_white", font=dict(size=12),
                                      xaxis_title="Slippage (BPS)", legend_title="Strategy")
                hist_path = f"./images/{symbol}_{period}_{direction}_slippage_hist.png"
                fig_hist.write_image(hist_path, scale=2)

            if return_dist:
                df_ret = pd.DataFrame(return_dist)
                fig_box = px.box(df_ret, x="Strategy", y="Value", points="all",
                                    title=f"{symbol} {period} {direction} Return Boxplot")
                fig_box.update_layout(width=700, height=350, template="simple_white", font=dict(size=12),
                                     xaxis_title="Strategy", yaxis_title="Return", legend_title="Strategy")
                box_path = f"./images/{symbol}_{period}_{direction}_return_boxplot.png"
                fig_box.write_image(box_path, scale=2)
                fig_violin = px.violin(df_ret, x="Strategy", y="Value", box=True, points="all",
                                        title=f"{symbol} {period} {direction} Return Violin")
                fig_violin.update_layout(width=700, height=350, template="simple_white", font=dict(size=12),
                                        xaxis_title="Strategy", yaxis_title="Return", legend_title="Strategy")
                violin_path = f"./images/{symbol}_{period}_{direction}_return_violin.png"
                fig_violin.write_image(violin_path, scale=2)
                fig_density = px.density_contour(df_ret, x="Value", color="Strategy",
                                                    title=f"{symbol} {period} {direction} Return Density")
                fig_density.update_layout(width=700, height=350, template="simple_white", font=dict(size=12),
                                         xaxis_title="Return", legend_title="Strategy")
                density_path = f"./images/{symbol}_{period}_{direction}_return_density.png"
                fig_density.write_image(density_path, scale=2)
                fig_ecdf = px.ecdf(df_ret, x="Value", color="Strategy",
                                    title=f"{symbol} {period} {direction} Return ECDF")
                fig_ecdf.update_layout(width=700, height=350, template="simple_white", font=dict(size=12),
                                      xaxis_title="Return", legend_title="Strategy")
                ecdf_path = f"./images/{symbol}_{period}_{direction}_return_ecdf.png"
                fig_ecdf.write_image(ecdf_path, scale=2)
                fig_hist = px.histogram(df_ret, x="Value", color="Strategy", barmode="overlay",
                                        nbins=50, marginal="violin",
                                        title=f"{symbol} {period} {direction} Return Histogram")
                fig_hist.update_layout(width=700, height=350, template="simple_white", font=dict(size=12),
                                      xaxis_title="Return", legend_title="Strategy")
                hist_path = f"./images/{symbol}_{period}_{direction}_return_hist.png"
                fig_hist.write_image(hist_path, scale=2)

            if retracement_dist:
                df_ret = pd.DataFrame(retracement_dist)
                fig_box = px.box(df_ret, x="Strategy", y="Value", points="all",
                                    title=f"{symbol} {period} {direction} Retracement Boxplot")
                fig_box.update_layout(width=700, height=350, template="simple_white", font=dict(size=12),
                                     xaxis_title="Strategy", yaxis_title="Retracement", legend_title="Strategy")
                box_path = f"./images/{symbol}_{period}_{direction}_retracement_boxplot.png"
                fig_box.write_image(box_path, scale=2)
                fig_violin = px.violin(df_ret, x="Strategy", y="Value", box=True, points="all",
                                        title=f"{symbol} {period} {direction} Retracement Violin")
                fig_violin.update_layout(width=700, height=350, template="simple_white", font=dict(size=12),
                                        xaxis_title="Strategy", yaxis_title="Retracement", legend_title="Strategy")
                violin_path = f"./images/{symbol}_{period}_{direction}_retracement_violin.png"
                fig_violin.write_image(violin_path, scale=2)
                fig_density = px.density_contour(df_ret, x="Value", color="Strategy",
                                                    title=f"{symbol} {period} {direction} Retracement Density")
                fig_density.update_layout(width=700, height=350, template="simple_white", font=dict(size=12),
                                         xaxis_title="Retracement", legend_title="Strategy")
                density_path = f"./images/{symbol}_{period}_{direction}_retracement_density.png"
                fig_density.write_image(density_path, scale=2)
                fig_ecdf = px.ecdf(df_ret, x="Value", color="Strategy",
                                    title=f"{symbol} {period} {direction} Retracement ECDF")
                fig_ecdf.update_layout(width=700, height=350, template="simple_white", font=dict(size=12),
                                      xaxis_title="Retracement", legend_title="Strategy")
                ecdf_path = f"./images/{symbol}_{period}_{direction}_retracement_ecdf.png"
                fig_ecdf.write_image(ecdf_path, scale=2)
                fig_hist = px.histogram(df_ret, x="Value", color="Strategy", barmode="overlay",
                                        nbins=50, marginal="violin",
                                        title=f"{symbol} {period} {direction} Retracement Histogram")
                fig_hist.update_layout(width=700, height=350, template="simple_white", font=dict(size=12),
                                      xaxis_title="Retracement", legend_title="Strategy")
                hist_path = f"./images/{symbol}_{period}_{direction}_retracement_hist.png"
                fig_hist.write_image(hist_path, scale=2)

            # VWAP comparison curves (unchanged)
            if period_symbol_df_pathes is not None and base_path is not None:
                allowed_windows = interval_to_windows.get(period, [1])
                for execution_window in allowed_windows:
                    df_files = period_symbol_df_pathes.get(period, {}).get(symbol, [])
                    if not df_files:
                        continue
                    strategies = ["kdj", "rsi", "boll", "dual_thrust", "macd", "GAN"]
                    for strategy in strategies:
                        dfs = {"transformer": None, "rnn": None, "lstm": None}
                        for df_file in df_files:
                            fname = os.path.basename(df_file)
                            if (
                                f"_{execution_window}MINexecution" in fname
                                and strategy in fname
                            ):
                                try:
                                    df = pd.read_csv(df_file, index_col=0, encoding='utf-8')
                                except Exception:
                                    continue
                                if "transformer" in fname:
                                    dfs["transformer"] = df
                                elif "rnn" in fname:
                                    dfs["rnn"] = df
                                elif "lstm" in fname:
                                    dfs["lstm"] = df
                        if any(df is not None for df in dfs.values()):
                            fig = go.Figure()
                            for model_name, df in dfs.items():
                                if df is not None and 'execution_price' in df.columns:
                                    fig.add_trace(go.Scatter(
                                        x=df.index,
                                        y=df['execution_price'],
                                        name=f'Actual Execution Price (HLC/3) [{model_name}]',
                                        line=dict(color='darkorange', width=1, dash='dot')
                                    ))
                                    break
                            for model_name, df in dfs.items():
                                if df is not None and 'traditional_vwap_line' in df.columns:
                                    fig.add_trace(go.Scatter(
                                        x=df.index,
                                        y=df['traditional_vwap_line'],
                                        name='Benchmark VWAP (HLC/3)',
                                        line=dict(color='royalblue', width=2)
                                    ))
                                    break
                            if dfs["transformer"] is not None and 'model_vwap_line' in dfs["transformer"].columns:
                                fig.add_trace(go.Scatter(
                                    x=dfs["transformer"].index,
                                    y=dfs["transformer"]['model_vwap_line'],
                                    name='Our-Transformer VWAP',
                                    line=dict(color='red', width=2, dash='dash')
                                ))
                            if dfs["rnn"] is not None and 'model_vwap_line' in dfs["rnn"].columns:
                                fig.add_trace(go.Scatter(
                                    x=dfs["rnn"].index,
                                    y=dfs["rnn"]['model_vwap_line'],
                                    name='Our-RNN VWAP',
                                    line=dict(color='green', width=2, dash='dot')
                                ))
                            if dfs["lstm"] is not None and 'model_vwap_line' in dfs["lstm"].columns:
                                fig.add_trace(go.Scatter(
                                    x=dfs["lstm"].index,
                                    y=dfs["lstm"]['model_vwap_line'],
                                    name='Our-LSTM VWAP',
                                    line=dict(color='orange', width=2, dash='longdash')
                                ))
                            if strategy == "GAN":
                                for trace in fig.data:
                                    if "MAA" in trace.name or "GAN" in trace.name or "Our-Transformer VWAP" in trace.name or "Our-RNN VWAP" in trace.name or "Our-LSTM VWAP" in trace.name:
                                        trace.line.width = 2
                                        trace.line.color = "magenta"
                                        trace.line.dash = None
                            fig.update_layout(
                                title=f"{symbol} {execution_window}MIN {period} {strategy} VWAP/Execution Price Comparison",
                                xaxis_title="Time", yaxis_title="Price",
                                hovermode="x unified", legend_title="Indicator",
                                template="simple_white", font=dict(size=13),
                                width=900, height=400, margin=dict(l=40, r=10, t=40, b=40)
                            )
                            os.makedirs(f".images//vwap//", exist_ok=True)
                            save_path = f".images//vwap//{symbol}_{strategy}_{execution_window}MIN_{period}.png"
                            fig.write_image(save_path, scale=2)

            # Heatmap
            if slippage_dist:
                df_slip = pd.DataFrame(slippage_dist)
                pivot = df_slip.pivot_table(index="Strategy", columns="Model", values="Value", aggfunc="mean")
                fig_heat = px.imshow(
                    pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    color_continuous_scale="RdBu_r",
                    labels=dict(x="Model", y="Strategy", color="Mean Slippage"),
                    title=f"{symbol} {period} {direction} Slippage Mean Heatmap"
                )
                fig_heat.update_layout(width=500, height=350, font=dict(size=12),
                                      xaxis_title="Model", yaxis_title="Strategy")
                heat_path = f"./images/{symbol}_{period}_{direction}_slippage_heatmap.png"
                fig_heat.write_image(heat_path, scale=2)

            # Kline + signal chart (unchanged)
            for strat, df in kline_signal_data:
                if not all(col in df.columns for col in ["date", "open", "high", "low", "close", "signal"]):
                    continue
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"])
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                    name="Kline", increasing_line_color='red', decreasing_line_color='green', showlegend=True
                ))
                buy_idx = df[df["signal"].isin(["Buy"])].index
                sell_idx = df[df["signal"].isin(["Sell"])].index
                close_buy_idx = df[df["signal"].isin(["Close_Buy"])].index
                close_sell_idx = df[df["signal"].isin(["Close_Sell"])].index
                fig.add_trace(go.Scatter(
                    x=df.loc[buy_idx, "date"], y=df.loc[buy_idx, "low"]*0.995,
                    mode="markers", marker=dict(symbol="triangle-up", color="blue", size=8),
                    name="Buy"
                ))
                fig.add_trace(go.Scatter(
                    x=df.loc[sell_idx, "date"], y=df.loc[sell_idx, "high"]*1.005,
                    mode="markers", marker=dict(symbol="triangle-down", color="orange", size=8),
                    name="Sell"
                ))
                fig.add_trace(go.Scatter(
                    x=df.loc[close_buy_idx, "date"], y=df.loc[close_buy_idx, "high"]*1.002,
                    mode="markers", marker=dict(symbol="star", color="purple", size=7),
                    name="Close Buy"
                ))
                fig.add_trace(go.Scatter(
                    x=df.loc[close_sell_idx, "date"], y=df.loc[close_sell_idx, "low"]*0.998,
                    mode="markers", marker=dict(symbol="star", color="black", size=7),
                    name="Close Sell"
                ))
                fig.update_layout(
                    title=f"{symbol} {period} {direction} {strat} Kline+Signal",
                    xaxis_title="Time", yaxis_title="Price",
                    template="simple_white", width=1200, height=500,
                    font=dict(size=13), margin=dict(l=40, r=10, t=40, b=40)
                )
                os.makedirs(f"./images/kline/", exist_ok=True)
                kline_path = f"./images/kline/{symbol}_{period}_{direction}_{strat}_kline_signal.png"
                fig.write_image(kline_path, scale=2)

            # All strategies on one Kline+signal chart (unchanged)
            if kline_signal_data:
                base_df = None
                for _, df in kline_signal_data:
                    if all(col in df.columns for col in ["date", "open", "high", "low", "close"]):
                        base_df = df.copy()
                        break
                if base_df is not None:
                    base_df["date"] = pd.to_datetime(base_df["date"])
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=base_df["date"], open=base_df["open"], high=base_df["high"], low=base_df["low"], close=base_df["close"],
                        name="Kline", increasing_line_color='red', decreasing_line_color='green', showlegend=True
                    ))
                    strat_marker_map = {
                        "MACD": dict(color="#636EFA", symbol="triangle-up", size=7),
                        "BOLL": dict(color="#EF553B", symbol="triangle-up", size=7),
                        "RSI": dict(color="#00CC96", symbol="triangle-up", size=7),
                        "Dual Thrust": dict(color="#AB63FA", symbol="triangle-up", size=7),
                        "KDJ": dict(color="#FFA15A", symbol="triangle-up", size=7),
                        "MAA": dict(color="#19D3F3", symbol="triangle-up", size=7),
                    }
                    strat_marker_map_sell = {
                        "MACD": dict(color="#636EFA", symbol="triangle-down", size=7),
                        "BOLL": dict(color="#EF553B", symbol="triangle-down", size=7),
                        "RSI": dict(color="#00CC96", symbol="triangle-down", size=7),
                        "Dual Thrust": dict(color="#AB63FA", symbol="triangle-down", size=7),
                        "KDJ": dict(color="#FFA15A", symbol="triangle-down", size=7),
                        "MAA": dict(color="#19D3F3", symbol="triangle-down", size=7),
                    }
                    for strat, df in kline_signal_data:
                        if not all(col in df.columns for col in ["date", "low", "high", "signal"]):
                            continue
                        df = df.copy()
                        df["date"] = pd.to_datetime(df["date"])
                        buy_idx = df[df["signal"].isin(["Buy"])].index
                        sell_idx = df[df["signal"].isin(["Sell"])].index
                        close_buy_idx = df[df["signal"].isin(["Close_Buy"])].index
                        close_sell_idx = df[df["signal"].isin(["Close_Sell"])].index
                        fig.add_trace(go.Scatter(
                            x=df.loc[buy_idx, "date"], y=df.loc[buy_idx, "low"]*0.995,
                            mode="markers",
                            marker=strat_marker_map.get(strat, dict(color="blue", symbol="triangle-up", size=7)),
                            name=f"{strat} Buy"
                        ))
                        fig.add_trace(go.Scatter(
                            x=df.loc[sell_idx, "date"], y=df.loc[sell_idx, "high"]*1.005,
                            mode="markers",
                            marker=strat_marker_map_sell.get(strat, dict(color="orange", symbol="triangle-down", size=7)),
                            name=f"{strat} Sell"
                        ))
                        fig.add_trace(go.Scatter(
                            x=df.loc[close_buy_idx, "date"], y=df.loc[close_buy_idx, "high"]*1.002,
                            mode="markers",
                            marker=dict(symbol="star", color=strat_marker_map.get(strat, dict(color="purple"))["color"], size=6),
                            name=f"{strat} Close Buy"
                        ))
                        fig.add_trace(go.Scatter(
                            x=df.loc[close_sell_idx, "date"], y=df.loc[close_sell_idx, "low"]*0.998,
                            mode="markers",
                            marker=dict(symbol="star", color=strat_marker_map_sell.get(strat, dict(color="black"))["color"], size=6),
                            name=f"{strat} Close Sell"
                        ))
                    fig.update_layout(
                        title=f"{symbol} {period} {direction} All Strategies Kline+Signal",
                        xaxis_title="Time", yaxis_title="Price",
                        template="simple_white", width=1400, height=600,
                        font=dict(size=13), margin=dict(l=40, r=10, t=40, b=40)
                    )
                    os.makedirs(f"./images/kline/", exist_ok=True)
                    kline_path = f"./images/kline/{symbol}_{period}_{direction}_ALLSTRATEGY_kline_signal.png"
                    fig.write_image(kline_path, scale=2)

    print("All processing completed. Check the 'images' folder for results.")


for base_path in [r"D:\Projects\pycharm\VWAP2\VWAPCODES\buy_results",
                  "D:\Projects\pycharm\VWAP2\VWAPCODES\sell_results"]:
    file_pathes = []
    # 遍历base_path文件夹里所有文件和文件夹
    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            if ".csv" in file_path and "aanew" in file_path:
                file_pathes.append(file_path)

        for dir in dirs:
            # print(f"Processing directory: {dir}")
            dir_path = os.path.join(root, dir)
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if ".csv" in file_path and "aanew" in file_path:
                        # file_path = file_path.split(".csv")[0] + "_"+ dir + ".csv"  # 确保是.csv结尾
                        print(f"Processing file: {file_path}")
                        file_pathes.append(file_path)

    # 提取所有品种
    period_pathes = {}
    for period in periods:
        if period not in period_pathes:
            period_pathes[period] = []

    for period in periods:
        for file_path in file_pathes:
            # 提取所有 period 出现的位置和内容
            file_path_cur = file_path.split("1min_data")[0]
            if period not in file_path_cur:
                continue
            period_pathes[period].append(file_path)

    period_symbol_pathes = {}
    period_symbol_metrics_pathes = {}
    period_symbol_df_pathes = {}
    for period in periods:
        period_symbol_pathes[period] = {}
        period_symbol_metrics_pathes[period] = {}
        period_symbol_df_pathes[period] = {}
        # 初始化每个品种的路径列表
        for symbol in symbols:
            period_symbol_pathes[period][symbol] = []
            period_symbol_metrics_pathes[period][symbol] = []
            period_symbol_df_pathes[period][symbol] = []

    for period, file_pathes in period_pathes.items():
        for symbol in symbols:
            symbol_pathes = []
            for file_path in file_pathes:
                if symbol in file_path:  # 单个品种
                    symbol_pathes.append(file_path)

            if len(symbol_pathes) == 0:
                continue

            for file_path in symbol_pathes:
                if "metrics" in file_path:
                    period_symbol_metrics_pathes[period][symbol].append(file_path)
                if "df" in file_path:
                    period_symbol_df_pathes[period][symbol].append(file_path)
            period_symbol_pathes[period][symbol].append(symbol_pathes)

    period_symbol_metrics_window_pathes = {}
    period_symbol_df_window_pathes = {}
    for period in periods:
        period_symbol_metrics_window_pathes[period] = {}
        period_symbol_df_window_pathes[period] = {}
        # 初始化每个品种的路径列表
        for symbol in symbols:
            period_symbol_metrics_window_pathes[period][symbol] = {}
            period_symbol_df_window_pathes[period][symbol] = {}
            windows = periods_windows[period]
            for window in windows:
                period_symbol_metrics_window_pathes[period][symbol][window] = []
                period_symbol_df_window_pathes[period][symbol][window] = []

    for period in periods:
        for symbol in symbols:
            windows = periods_windows[period]
            for window in windows:
                for file_path in period_symbol_metrics_pathes[period][symbol]:
                    if str(window) in file_path:
                        period_symbol_metrics_window_pathes[period][symbol][window].append(file_path)
                for file_path in period_symbol_df_pathes[period][symbol]:
                    if str(window) in file_path:
                        period_symbol_df_window_pathes[period][symbol][window].append(file_path)
for symbol in symbols:
    direction = "buy" if "buy" in base_path else "sell"
    print(f"Processing symbol: {symbol}, direction: {direction}")
    print(f"Building table for symbol: {symbol}")
    # 需要依次调用函数，按顺序调用一个另外两个注释。
    # 第一个sheet
    # build_table_data2(direction, symbol, period_symbol_metrics_window_pathes, period_symbol_df_window_pathes, period_symbol_pathes)
    # 第二个sheet
    build_table_data33(direction, symbol, period_symbol_metrics_pathes, period_symbol_df_pathes, period_symbol_pathes)
    # 画所有指标曲线
    plot_all_metrics_curves(symbol, periods, period_symbol_metrics_pathes, direction, period_symbol_df_pathes,
                            base_path)