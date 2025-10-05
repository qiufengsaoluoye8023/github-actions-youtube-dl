#!/usr/bin/env python3
import argparse
import os
import sys
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ----------------------
# Utilities
# ----------------------

def compute_rsi(series: pd.Series, period: int = 8) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi


def train_hmm(
    X: np.ndarray,
    n_states: int = 5,
    cov_type: str = "full",
    n_init: int = 10,
    n_iter: int = 1000,
    tol: float = 1e-6,
    init_model: Optional[GaussianHMM] = None,
) -> Tuple[GaussianHMM, np.ndarray, float]:
    best_model: Optional[GaussianHMM] = None
    best_score: float = -np.inf
    best_states_seq: Optional[np.ndarray] = None

    for _ in range(n_init):
        if init_model is not None:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=cov_type,
                n_iter=n_iter,
                init_params="",
                tol=tol,
                verbose=False,
            )
            model.startprob_ = init_model.startprob_.copy()
            model.transmat_ = init_model.transmat_.copy()
            model.means_ = init_model.means_.copy()
            model.covars_ = init_model.covars_.copy()
        else:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=cov_type,
                n_iter=n_iter,
                init_params="kmeans",
                tol=tol,
                verbose=False,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)
        score = model.score(X)
        if score > best_score:
            best_model, best_score, best_states_seq = model, score, model.predict(X)

    assert best_model is not None and best_states_seq is not None
    return best_model, best_states_seq, best_score


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_features(
    df: pd.DataFrame,
    rsi_period: int,
    rolling_vol_period: int,
    ema_short: int,
    ema_long: int,
) -> pd.DataFrame:
    log_return = np.log(df["close"] / df["close"].shift(1))
    # clip extreme returns to reduce outliers impact
    low = np.nanpercentile(log_return, 1)
    high = np.nanpercentile(log_return, 99)
    log_return = np.clip(log_return, low, high)

    rolling_vol = log_return.rolling(rolling_vol_period).std()
    range_val = df["high"] - df["low"]
    rsi_short = compute_rsi(df["close"], rsi_period)
    ema_short = df["close"].ewm(span=ema_short).mean()
    ema_long = df["close"].ewm(span=ema_long).mean()
    ema_diff = ema_short - ema_long

    features = (
        pd.DataFrame(
            {
                "log_return": log_return,
                "rolling_vol": rolling_vol,
                "range": range_val,
                "rsi_short": rsi_short,
                "ema_diff": ema_diff,
            }
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )
    return features


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train Gaussian HMM on market features with sliding window.")
    parser.add_argument("--file", required=True, help="Input CSV path containing at least columns: time, open, high, low, close")
    parser.add_argument("--plot-dir", default="HMM_PLOTS", help="Output directory for plots and CSV summaries")
    parser.add_argument("--output", default="hmm_full_vs_sliding_window_outputs_5states.csv", help="Main output CSV path")

    parser.add_argument("--rsi-period", type=int, default=8)
    parser.add_argument("--rolling-vol-period", type=int, default=20)
    parser.add_argument("--ema-short", type=int, default=5)
    parser.add_argument("--ema-long", type=int, default=20)

    parser.add_argument("--n-states", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=2000)
    parser.add_argument("--retrain-freq", type=int, default=100)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--ewma-alpha", type=float, default=0.2)
    parser.add_argument("--min-window-size", type=int, default=50)

    parser.add_argument("--lstm-file", default="lstm_from_hmm_predictions_gpu_safe.csv")

    args = parser.parse_args(argv)

    ensure_dir(args.plot_dir)

    # 1) Load data
    df = pd.read_csv(args.file)
    required_cols = {"time", "high", "low", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: missing required columns: {sorted(missing)}", file=sys.stderr)
        return 2

    # 1. features
    features = build_features(
        df=df,
        rsi_period=args.rsi_period,
        rolling_vol_period=args.rolling_vol_period,
        ema_short=args.ema_short,
        ema_long=args.ema_long,
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # 2) Full-data HMM
    print("开始训练全数据 HMM...")
    full_model, full_states_seq, full_score = train_hmm(
        X_scaled,
        n_states=args.n_states,
        cov_type="full",
        n_init=args.n_init,
        n_iter=args.n_iter,
        tol=args.tol,
    )

    # BIC: -2 * log(L) + k * log(N)
    n_features = X_scaled.shape[1]
    n_states = args.n_states
    n_params = n_states * (n_states + 2 * n_features + n_states - 1)
    bic = -2 * full_score + n_params * np.log(X_scaled.shape[0])
    print(f"全数据 HMM 训练完成。Log-likelihood: {full_score:.2f}, BIC: {bic:.2f}")

    # 3) Initialize outputs
    n_samples = X_scaled.shape[0]
    output_df = df.loc[features.index].copy()
    output_df["log_return"] = features["log_return"]
    output_df["full_state_id"] = full_states_seq
    output_df["window_state_id"] = np.nan
    for k in range(n_states):
        output_df[f"window_state_prob_{k}"] = np.nan

    # 4) Sliding window
    print("开始进行滑动窗口 HMM 滚动训练...")
    init_model = full_model
    window_scores: List[float] = []

    for start in range(0, n_samples, args.retrain_freq):
        print(f"处理进度: {start / n_samples * 100:.2f}%")
        end = min(start + args.window_size, n_samples)
        X_window = X_scaled[start:end]
        if len(X_window) < args.min_window_size:
            break
        model, states_seq, score = train_hmm(
            X_window,
            n_states=n_states,
            cov_type="full",
            n_init=args.n_init,
            n_iter=args.n_iter,
            tol=args.tol,
            init_model=init_model,
        )
        window_scores.append(score)
        probs = pd.DataFrame(model.predict_proba(X_window)).ewm(alpha=args.ewma_alpha).mean()

        output_df.loc[features.index[start:end], "window_state_id"] = states_seq
        for k in range(n_states):
            output_df.loc[features.index[start:end], f"window_state_prob_{k}"] = probs[k].values

        init_model = model
    print("滑动窗口 HMM 滚动训练完成。")

    # 5) State duration
    output_df["window_state_id"] = output_df["window_state_id"].fillna(-1).astype(int)
    block_ids = output_df["window_state_id"].ne(output_df["window_state_id"].shift()).cumsum()
    output_df["window_state_duration"] = block_ids.map(
        block_ids.groupby(block_ids).transform("size")
    )
    output_df.loc[output_df["window_state_id"] == -1, ["window_state_id", "window_state_duration"]] = np.nan
    output_df["window_state_id"] = (
        output_df["window_state_id"].replace(np.nan, -1).astype("Int64").replace(-1, np.nan)
    )

    # 6) Plots & CSVs
    pd.DataFrame({"log_likelihood": window_scores}).to_csv(os.path.join(args.plot_dir, "1_log_likelihood_curve.csv"), index=False)
    plt.figure(figsize=(8, 3))
    plt.plot(window_scores)
    plt.title("Log-likelihood over windows")
    plt.tight_layout()
    plt.savefig(os.path.join(args.plot_dir, "1_log_likelihood_curve.png"))
    plt.close()

    output_df[["time", "close", "window_state_id"]].to_csv(os.path.join(args.plot_dir, "2_state_overlay.csv"), index=False)
    cols_probs = [f"window_state_prob_{k}" for k in range(n_states)]
    output_df[["time"] + cols_probs].to_csv(os.path.join(args.plot_dir, "3_state_probabilities.csv"), index=False)
    output_df[["time", "window_state_id", "window_state_duration"]].to_csv(os.path.join(args.plot_dir, "4_state_duration_histogram.csv"), index=False)

    # 8) State summary
    state_dist = (
        output_df["window_state_id"].value_counts(normalize=True).sort_index().rename("proportion").to_frame()
    )
    state_stats = output_df.groupby("window_state_id")["log_return"].agg(["mean", "std"]) if not output_df.empty else pd.DataFrame(columns=["mean","std"])
    median_std = state_stats["std"].median() if not state_stats.empty else 0.0

    summary_records = []
    for state_id in range(full_model.n_components):
        mean_vec = full_model.means_[state_id]
        state_prop = state_dist.loc[state_id, "proportion"] if state_id in state_dist.index else 0.0
        ret_mean = state_stats.loc[state_id, "mean"] if state_id in state_stats.index else 0.0
        ret_std = state_stats.loc[state_id, "std"] if state_id in state_stats.index else 0.0

        if ret_mean > 0 and ret_std < median_std:
            market_type = "上涨（稳定）"
        elif ret_mean > 0 and ret_std >= median_std:
            market_type = "上涨（波动）"
        elif ret_mean < 0 and ret_std >= median_std:
            market_type = "下跌（剧烈）"
        elif ret_mean < 0 and ret_std < median_std:
            market_type = "下跌（平稳）"
        else:
            market_type = "震荡"

        summary_records.append(
            {
                "state_id": state_id,
                "proportion": state_prop,
                "mean_return": ret_mean,
                "volatility": ret_std,
                "market_type": market_type,
                "log_return_mean": float(mean_vec[0]),
                "rolling_vol_mean": float(mean_vec[1]),
                "range_mean": float(mean_vec[2]),
                "RSI_mean": float(mean_vec[3]),
                "EMA_diff_mean": float(mean_vec[4]),
            }
        )

    df_summary = pd.DataFrame(summary_records).sort_values("mean_return", ascending=False)
    summary_path = os.path.join(args.plot_dir, "state_summary.csv")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 状态特征汇总表已保存: {summary_path}")

    # 9) Save outputs and LSTM input
    output_df.to_csv(args.output, index=False)
    print(f"✅ 主结果已保存到: {args.output}")

    lstm_cols = [
        "time",
        "log_return",
        "window_state_id",
        "window_state_duration",
    ] + [f"window_state_prob_{k}" for k in range(n_states)]
    lstm_input_df = output_df[lstm_cols].copy()
    lstm_input_file = os.path.join(os.path.dirname(args.output) or ".", args.lstm_file)
    lstm_input_df.to_csv(lstm_input_file, index=False, encoding="utf-8-sig")
    print(f"✅ LSTM 输入文件已保存到: {lstm_input_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
