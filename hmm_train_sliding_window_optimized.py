import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

# ======================
# 配置参数
# ======================
CONFIG = {
    'file_path': r'C:\Users\Administrator\Desktop\MT5\XAUUSD_M5_last5years.csv',
    'features': ['log_return', 'rolling_vol', 'range', 'rsi_short', 'ema_diff'],
    'rsi_period': 8,
    'rolling_vol_period': 20,
    'ema_short': 5,
    'ema_long': 20,
    'n_states': 5,
    'window_size': 2000,
    'retrain_freq': 100,
    'n_init': 10,
    'n_iter': 1000,
    'tol': 1e-6,
    'ewma_alpha': 0.2,
    'min_window_size': 50,
    'output_file': 'hmm_full_vs_sliding_window_outputs_5states.csv',
    'plot_dir': 'HMM_PLOTS',
    # 新增的稳定/收敛相关参数
    'cov_type': 'diag',          # 建议先用 diag 提升稳定性
    'random_state': 42,          # 可复现
    'sticky_self_p': 0.97,       # 粘性自循环概率
    'min_covar': 1e-4            # 协方差下限，防退化
}

# ======================
# 确保输出目录存在
# ======================
os.makedirs(CONFIG['plot_dir'], exist_ok=True)

# ======================
# RSI 计算函数
# ======================
def compute_rsi(series, period=8):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)

# ======================
# 粘性转移矩阵 & KMeans 初始化
# ======================
def build_sticky_transmat(n_states: int, self_p: float) -> np.ndarray:
    off = (1.0 - self_p) / (n_states - 1)
    A = np.full((n_states, n_states), off, dtype=float)
    np.fill_diagonal(A, self_p)
    return A


def kmeans_init_params(X: np.ndarray, n_states: int, cov_type: str, random_state: int | None, min_covar: float):
    kmeans = KMeans(n_clusters=n_states, n_init=20, random_state=random_state)
    labels = kmeans.fit_predict(X)
    means = kmeans.cluster_centers_
    n_features = X.shape[1]

    if cov_type == 'diag':
        covars = np.zeros((n_states, n_features), dtype=float)
        global_var = X.var(axis=0) + min_covar
        for k in range(n_states):
            cluster = X[labels == k]
            if cluster.shape[0] >= 2:
                covars[k] = cluster.var(axis=0) + min_covar
            else:
                covars[k] = global_var
        return means, covars

    covars = np.zeros((n_states, n_features, n_features), dtype=float)
    global_cov = np.cov(X.T) + np.eye(n_features) * min_covar
    for k in range(n_states):
        cluster = X[labels == k]
        if cluster.shape[0] >= 2:
            S = np.cov(cluster.T)
            covars[k] = S + np.eye(n_features) * min_covar
        else:
            covars[k] = global_cov
    return means, covars

# ======================
# HMM 训练函数 - 强化初始化以提升收敛率
# ======================
def train_hmm(
    X,
    n_states=5,
    cov_type='diag',
    n_init=10,
    n_iter=1000,
    tol=1e-6,
    init_model=None,
    random_state=42,
    sticky_self_p=0.97,
    min_covar=1e-4
):
    best_model, best_score, best_states_seq = None, -np.inf, None

    for i in range(n_init):
        if init_model is not None and i == 0:
            # 复用上一窗口/全量模型作为强初始化
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=cov_type,
                n_iter=n_iter,
                init_params='',   # 不重置参数
                tol=tol,
                random_state=random_state,
                verbose=False
            )
            model.startprob_ = init_model.startprob_.copy()
            model.transmat_ = init_model.transmat_.copy()
            model.means_ = init_model.means_.copy()
            model.covars_ = init_model.covars_.copy()
        else:
            # KMeans + 粘性转移矩阵初始化
            seed = None if random_state is None else random_state + i
            means, covars = kmeans_init_params(X, n_states, cov_type, seed, min_covar)
            startprob = np.full(n_states, 1.0 / n_states, dtype=float)
            transmat = build_sticky_transmat(n_states, sticky_self_p)

            model = GaussianHMM(
                n_components=n_states,
                covariance_type=cov_type,
                n_iter=n_iter,
                init_params='',   # 全部参数由我们提供
                tol=tol,
                random_state=seed,
                verbose=False
            )
            model.startprob_ = startprob
            model.transmat_ = transmat
            model.means_ = means
            model.covars_ = covars

        model.fit(X)
        score = model.score(X)
        if score > best_score:
            best_model, best_score = model, score
            best_states_seq = model.predict(X)

    return best_model, best_states_seq, best_score


# ======================
# 1. 加载数据与计算特征
# ======================
df = pd.read_csv(CONFIG['file_path'])

log_return = np.log(df['close'] / df['close'].shift(1))
log_return = np.clip(log_return, np.nanpercentile(log_return, 1), np.nanpercentile(log_return, 99))
rolling_vol = log_return.rolling(CONFIG['rolling_vol_period']).std()
range_val = df['high'] - df['low']
range_val = np.clip(range_val, np.nanpercentile(range_val, 1), np.nanpercentile(range_val, 99))
rsi_short = compute_rsi(df['close'], CONFIG['rsi_period'])
ema_short = df['close'].ewm(span=CONFIG['ema_short']).mean()
ema_long = df['close'].ewm(span=CONFIG['ema_long']).mean()
ema_diff = ema_short - ema_long
ema_diff = np.clip(ema_diff, np.nanpercentile(ema_diff, 1), np.nanpercentile(ema_diff, 99))

features = pd.DataFrame({
    'log_return': log_return,
    'rolling_vol': rolling_vol,
    'range': range_val,
    'rsi_short': rsi_short,
    'ema_diff': ema_diff
}).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

X_scaled = StandardScaler().fit_transform(features)

# ======================
# 2. 全数据 HMM 训练
# ======================
print("开始训练全数据 HMM...")
full_model, full_states_seq, full_score = train_hmm(
    X_scaled,
    n_states=CONFIG['n_states'],
    cov_type=CONFIG['cov_type'],
    n_init=CONFIG['n_init'],
    n_iter=CONFIG['n_iter'],
    tol=CONFIG['tol'],
    init_model=None,
    random_state=CONFIG['random_state'],
    sticky_self_p=CONFIG['sticky_self_p'],
    min_covar=CONFIG['min_covar']
)

# BIC（粗略近似）
n_params = CONFIG['n_states'] * (CONFIG['n_states'] + 2 * X_scaled.shape[1] + CONFIG['n_states'] - 1)
bic = -2 * full_score + n_params * np.log(X_scaled.shape[0])
print(f"全数据 HMM 训练完成。Log-likelihood: {full_score:.2f}, BIC: {bic:.2f}")

# ======================
# 3. 输出表初始化
# ======================
n_samples = X_scaled.shape[0]
output_df = df.loc[features.index].copy()
output_df['log_return'] = features['log_return']
output_df['full_state_id'] = full_states_seq
output_df['window_state_id'] = np.nan
for k in range(CONFIG['n_states']):
    output_df[f'window_state_prob_{k}'] = np.nan

# ======================
# 4. 滑动窗口训练
# ======================
print("开始进行滑动窗口 HMM 滚动训练...")
init_model = full_model
window_scores = []
for start in range(0, n_samples, CONFIG['retrain_freq']):
    print(f"处理进度: {start / n_samples * 100:.2f}%")
    end = min(start + CONFIG['window_size'], n_samples)
    X_window = X_scaled[start:end]
    if len(X_window) < CONFIG['min_window_size']:
        break

    model, states_seq, score = train_hmm(
        X_window,
        n_states=CONFIG['n_states'],
        cov_type=CONFIG['cov_type'],
        n_init=CONFIG['n_init'],
        n_iter=CONFIG['n_iter'],
        tol=CONFIG['tol'],
        init_model=init_model,
        random_state=CONFIG['random_state'],
        sticky_self_p=CONFIG['sticky_self_p'],
        min_covar=CONFIG['min_covar']
    )
    window_scores.append(score)

    # 概率 + EWMA 平滑
    probs = pd.DataFrame(model.predict_proba(X_window)).ewm(alpha=CONFIG['ewma_alpha']).mean()

    # 更新 output_df
    output_df.loc[features.index[start:end], 'window_state_id'] = states_seq
    for k in range(CONFIG['n_states']):
        output_df.loc[features.index[start:end], f'window_state_prob_{k}'] = probs[k].values

    # 下个窗口的初始化模型
    init_model = model
print("滑动窗口 HMM 滚动训练完成。")

# ======================
# 5. 状态持续时间
# ======================
output_df['window_state_id'] = output_df['window_state_id'].fillna(-1).astype(int)
output_df['window_state_duration'] = (
    output_df['window_state_id'].ne(output_df['window_state_id'].shift())
    .cumsum()
    .map(output_df.groupby(output_df['window_state_id'].ne(output_df['window_state_id'].shift()).cumsum())['window_state_id'].transform('size'))
)
output_df.loc[output_df['window_state_id'] == -1, ['window_state_id', 'window_state_duration']] = np.nan
output_df['window_state_id'] = output_df['window_state_id'].replace(np.nan, -1).astype('Int64').replace(-1, np.nan)

# ======================
# 6~8. 图表 + 状态统计 + 特征汇总
# ======================
plot_dir = CONFIG['plot_dir']

# 6. log-likelihood 曲线
pd.DataFrame({'log_likelihood': window_scores}).to_csv(os.path.join(plot_dir, '1_log_likelihood_curve.csv'), index=False)
plt.figure(figsize=(8, 3))
plt.plot(window_scores)
plt.title("Log-likelihood over windows")
plt.savefig(os.path.join(plot_dir, '1_log_likelihood_curve.png'))
plt.close()

# 7. 导出状态数据
output_df[['time', 'close', 'window_state_id']].to_csv(os.path.join(plot_dir, '2_state_overlay.csv'), index=False)
cols_probs = [f'window_state_prob_{k}' for k in range(CONFIG['n_states'])]
output_df[['time'] + cols_probs].to_csv(os.path.join(plot_dir, '3_state_probabilities.csv'), index=False)
output_df[['time', 'window_state_id', 'window_state_duration']].to_csv(os.path.join(plot_dir, '4_state_duration_histogram.csv'), index=False)

# 8. 状态特征汇总
state_dist = output_df['window_state_id'].value_counts(normalize=True).sort_index().rename('proportion').to_frame()
state_stats = output_df.groupby('window_state_id')['log_return'].agg(['mean', 'std'])

median_std = state_stats['std'].median() if not state_stats.empty else 0

summary_records = []
for state_id in range(full_model.n_components):
    mean_vec = full_model.means_[state_id]

    state_prop = state_dist.loc[state_id, 'proportion'] if state_id in state_dist.index else 0
    ret_mean = state_stats.loc[state_id, 'mean'] if state_id in state_stats.index else 0
    ret_std = state_stats.loc[state_id, 'std'] if state_id in state_stats.index else 0

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

    summary_records.append({
        'state_id': state_id,
        'proportion': state_prop,
        'mean_return': ret_mean,
        'volatility': ret_std,
        'market_type': market_type,
        'log_return_mean': mean_vec[0],
        'rolling_vol_mean': mean_vec[1],
        'range_mean': mean_vec[2],
        'RSI_mean': mean_vec[3],
        'EMA_diff_mean': mean_vec[4]
    })

df_summary = pd.DataFrame(summary_records).sort_values('mean_return', ascending=False)
summary_path = os.path.join(CONFIG['plot_dir'], 'state_summary.csv')
df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 状态特征汇总表已保存: {summary_path}")

# ======================
# 9. 保存主输出 + LSTM 输入文件
# ======================
output_df.to_csv(CONFIG['output_file'], index=False)
print(f"✅ 主结果已保存到: {CONFIG['output_file']}")

lstm_cols = ['time', 'log_return', 'window_state_id', 'window_state_duration'] + \
            [f'window_state_prob_{k}' for k in range(CONFIG['n_states'])]
lstm_input_df = output_df[lstm_cols].copy()

lstm_input_file = 'lstm_from_hmm_predictions_gpu_safe.csv'
lstm_input_df.to_csv(lstm_input_file, index=False, encoding='utf-8-sig')
print(f"✅ LSTM 输入文件已保存到: {lstm_input_file}")
