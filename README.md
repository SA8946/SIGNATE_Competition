# SIGNATE_Competition
SIGNATEコンペ参加時の機械学習実装コード
from google.colab import drive
drive.mount('/content/drive')

/content/drive/MyDrive

# ライブラリのインポート
import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import datetime as dt

from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# 日本語フォントを簡単に使う
!pip -q install japanize-matplotlib
import japanize_matplotlib

data_path = '/content/drive/MyDrive/data/'
# data_path = '/content/drive/Shareddrives/株式会社SIGNATE_プロジェクト/SIGNATE Competition/20.コンペ成果物/20.オープンコンペ/SMBC Group/SMBC Group（2025）/20.データ/open/'

train_df  = pd.read_csv(data_path + 'train.csv', index_col=0)
test_df   = pd.read_csv(data_path + 'test.csv', index_col=0)
sample_submission_df = pd.read_csv(data_path + 'sample_submit.csv', header=None)  # header 無し

feature_desc_df = pd.read_csv(data_path + 'feature_description.csv')

feature_desc_df

# 概要を確認
print("\n--- 学習データ (train_df) ---")
display(train_df.head())
print(f"shape: {train_df.shape}")

# print("\n--- テストデータ (test_df) ---")
# display(test_df.head())
# print(f"shape: {test_df.shape}")

# print("\n--- サンプル提出 (sample_submission_df) ---")
# display(sample_submission_df.head())
# print(f"shape: {sample_submission_df.shape}")

# 情報や統計量を確認
print("\n--- 学習データ (train_df) ---")
display(train_df.info())
display(train_df.describe())
print()
print("\n--- テストデータ (test_df) ---")
display(test_df.info())
display(test_df.describe())

base_features = [
    'total_load_actual',
    'generation_wind_onshore',
    'generation_hydro_pumped_storage_consumption',
    'generation_hydro_run_of_river_and_poundage',
    'generation_hydro_water_reservoir',
    'generation_fossil_gas',
    'generation_fossil_hard_coal',
    'generation_fossil_brown_coal/lignite',
    'valencia_weather_main',
    'madrid_weather_main',
    'bilbao_weather_main',
    'barcelona_weather_main',
    'seville_weather_main',
    'valencia_temp',
    'price_actual'
]

train_sub = train_df[base_features].copy()
test_sub = test_df[base_features[:-1]].copy() #testデータでは目的変数以外の特徴量を選択

corr = train_sub.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("価格と主要変数の相関")
plt.show()
print()
abs_corr = corr["price_actual"].drop("price_actual").abs().sort_values(ascending=True)
abs_corr.plot(kind="barh", figsize=(12,10))
plt.title("価格との相関（絶対値）")
plt.xlabel("|corr|")
plt.show()


# 目的変数と特に相関が高い3つの特徴量の散布図を可視化
top3 = abs_corr.sort_values(ascending=False).head(3).index.tolist()

for col in top3:
    plt.figure(figsize=(8,8))
    sns.scatterplot(x=train_sub[col], y=train_sub["price_actual"], alpha=0.3)
    sns.regplot(x=train_sub[col], y=train_sub["price_actual"],
                scatter=False, color="red")
    plt.title(f"price_actual vs {col}")
    plt.show()


COMFORT = 20.0 # 快適温度 ≈ 20 ℃ と仮定
train_sub["temp_C"] = train_sub["valencia_temp"] - 273.15 # ケルビンから摂氏に変換
train_sub["temp_dev"] = (train_sub["temp_C"] - COMFORT).abs() # 20℃からどれぐらい離れているかを計算しtemp_devという特徴量を作成

# testデータにも同様の特徴量を追加
test_sub["temp_C"] = test_sub["valencia_temp"] - 273.15
test_sub["temp_dev"] = (test_sub["temp_C"] - COMFORT).abs()


plt.figure(figsize=(8,8))
sns.scatterplot(x=train_sub["temp_dev"], y=train_sub["price_actual"], alpha=0.3)
sns.regplot(x=train_sub["temp_dev"], y=train_sub["price_actual"],
            scatter=False, color="red")
plt.xlabel("|気温−20℃|")
plt.ylabel("price_actual")
plt.title("電力価格 vs 気温偏差")
plt.show()


corr_temp = train_sub["temp_dev"].corr(train_sub["price_actual"])
print(f"気温偏差と需要の相関: {corr_temp:.4f}")


# トレーニング用データの時間軸データ前処理
## インデックスを datetime に変換（タイムゾーンあり）
train_sub.index = pd.to_datetime(train_sub.index, utc=True)

## datetimeを列として追加
train_sub["datetime"] = train_sub.index

## サマータイムを考慮してローカライズ
train_sub["datetime"] = train_sub["datetime"].dt.tz_convert('Europe/Madrid')

## タイムゾーンを除去してNaiveなdatetimeに変換
train_sub["datetime"] = train_sub["datetime"].dt.tz_localize(None)

## 特徴量を抽出（インデックスから直接でも、datetime列からでもOK）
train_sub["month"] = train_sub["datetime"].dt.month
train_sub["day"] = train_sub["datetime"].dt.day
train_sub["hour"] = train_sub["datetime"].dt.hour
train_sub["weekday"] = train_sub["datetime"].dt.weekday


# テストデータにも同じ時間軸データ前処理
## インデックスを datetime に変換（タイムゾーンあり）
test_sub.index = pd.to_datetime(test_sub.index, utc=True)

## datetimeを列として追加
test_sub["datetime"] = test_sub.index

## サマータイムを考慮してローカライズ
test_sub["datetime"] = test_sub["datetime"].dt.tz_convert('Europe/Madrid')

## タイムゾーンを除去してNaiveなdatetimeに変換
test_sub["datetime"] = test_sub["datetime"].dt.tz_localize(None)

## 特徴量を抽出（インデックスから直接でも、datetime列からでもOK）
test_sub["month"] = test_sub["datetime"].dt.month
test_sub["day"] = test_sub["datetime"].dt.day
test_sub["hour"] = test_sub["datetime"].dt.hour
test_sub["weekday"] = test_sub["datetime"].dt.weekday


plt.figure(figsize=(30,8))
plt.plot(test_sub["datetime"], test_sub["total_load_actual"])
plt.show()

value_counts_seville = test_sub['valencia_weather_main'].value_counts()
value_counts_madrid = test_sub['madrid_weather_main'].value_counts()
value_counts_bilbao = test_sub['bilbao_weather_main'].value_counts()
value_counts_barcelona = test_sub['barcelona_weather_main'].value_counts()
value_counts_seville = test_sub['seville_weather_main'].value_counts()

print(value_counts_seville)
print(value_counts_madrid)
print(value_counts_bilbao)
print(value_counts_barcelona)
print(value_counts_seville)

display(train_sub.info())

# 文字列の天気情報を0から1の範囲にマッピングする辞書作成
w_dic = {'clear':0.0, 'clouds':0.5, 'rain':1.0, 'thunderstorm':1.0, 'mist':0.6, 'drizzle':0.7, 'fog':0.7, 'snow':1.0, 'squall':1.0, 'haze':0.6, 'smoke':0.6}

# train_subの文字列を辞書を使って変換
train_sub['valencia_weather_main'] = train_sub['valencia_weather_main'].map(w_dic)
train_sub['madrid_weather_main'] = train_sub['madrid_weather_main'].map(w_dic)
train_sub['bilbao_weather_main'] = train_sub['bilbao_weather_main'].map(w_dic)
train_sub['barcelona_weather_main'] = train_sub['barcelona_weather_main'].map(w_dic)
train_sub['seville_weather_main'] = train_sub['seville_weather_main'].map(w_dic)

# test_subの文字列を辞書を使って変換
test_sub['valencia_weather_main'] = test_sub['valencia_weather_main'].map(w_dic)
test_sub['madrid_weather_main'] = test_sub['madrid_weather_main'].map(w_dic)
test_sub['bilbao_weather_main'] = test_sub['bilbao_weather_main'].map(w_dic)
test_sub['barcelona_weather_main'] = test_sub['barcelona_weather_main'].map(w_dic)
test_sub['seville_weather_main'] = test_sub['seville_weather_main'].map(w_dic)


train_sub['valencia_weather_main']

select_col = [
    'total_load_actual',
    'month',
    'day',
    'hour',
    'weekday',
    'generation_wind_onshore',
    'generation_hydro_pumped_storage_consumption',
    'generation_hydro_run_of_river_and_poundage',
    'generation_hydro_water_reservoir',
    'generation_fossil_gas',
    'generation_fossil_hard_coal',
    'generation_fossil_brown_coal/lignite',
    'temp_dev',
    'valencia_weather_main',
    'madrid_weather_main',
    'bilbao_weather_main',
    'barcelona_weather_main',
    'seville_weather_main',
    'price_actual'
]
train_selected = train_sub[select_col]
test_selected = test_sub[select_col[:-1]]


train_selected = train_selected.astype(np.float32)
train_selected

test_selected

# 欠損値補完
train_selected.ffill(inplace=True)
test_selected.ffill(inplace=True)

# 目的変数以外のカラムを標準化
# scale_cols = [c for c in train_selected.columns if c != "price_actual"]
# 目的変数も含めて全カラムを標準化対象に
scale_cols = list(train_selected.columns)  # 全てのカラムを含める（price_actual も含む）

# OK / NGをそれぞれ別のデータフレームで作成
train_ok = train_selected.copy()
train_ng = train_selected.copy()
test_ok = test_selected.copy()
test_ng = test_selected.copy()

# train データ：StandardScaler で一括標準化
scaler = StandardScaler()
scaled_arr = scaler.fit_transform(train_ok[scale_cols])
for i, col in enumerate(scale_cols):
    train_ok[f"{col}_scaled"] = scaled_arr[:, i]          # ← index はそのまま

# test データ：逐次標準化
# 時系列順に連結
df_all = (pd.concat([train_ok.assign(dataset='Train'),
                     test_ok.assign(dataset='Test')])
          .sort_index())

# expanding() で「その行まで」の平均・標準偏差
for col in scale_cols:
    df_all[f'{col}_mean_to_t'] = df_all[col].expanding().mean()
    df_all[f'{col}_std_to_t']  = df_all[col].expanding().std(ddof=0)

    mask_test = df_all['dataset'] == 'Test'
    df_all.loc[mask_test, f'{col}_scaled'] = (
        (df_all.loc[mask_test, col] - df_all.loc[mask_test, f'{col}_mean_to_t']) /
        df_all.loc[mask_test, f'{col}_std_to_t'].replace(0, np.nan)
    )

# train / test に切り戻す
cols_to_drop = (['dataset']
                + [f'{c}_mean_to_t' for c in scale_cols]
                + [f'{c}_std_to_t'  for c in scale_cols])

train_ok = df_all[df_all['dataset'] == 'Train'].drop(columns=cols_to_drop).copy()
test_ok  = df_all[df_all['dataset'] == 'Test' ].drop(columns=cols_to_drop).copy()

np.isnan(test_ok).sum(), np.isinf(test_ok).sum()

display(train_ok.info())
display(test_ok.info())


print("◆ OK（リークなし）")
display(test_ok[[*(f"{c}_scaled" for c in scale_cols) ]].head())

print("\n◆ NG（未来情報で標準化：参考）")
display(test_ng[[*(f"{c}_scaled" for c in scale_cols) ]].head())

train_ok


# yearで分割するためindexをDatetimeIndex化
train_ok.index = pd.to_datetime(train_ok.index, errors="coerce", utc=True)
train_ok.index = train_ok.index.tz_convert("Etc/GMT-1")

# 2017 年をバリデーションに分割
val_mask = train_ok.index.year == 2017
train_mask = ~val_mask

# 標準化した特徴量のみを選択
# feature_cols = [c for c in train_ok.columns if c.endswith("_scaled")]
feature_cols = [
    c for c in train_ok.columns
    if (c.endswith('_scaled') and c not in {'month_scaled', 'day_scaled', 'hour_scaled', 'weekday_scaled'})
    or c in {'month', 'day', 'hour', 'weekday'}
]

# y（price_actual）の標準化
scaler_y = StandardScaler()
train_ok['price_actual_scaled'] = scaler_y.fit_transform(train_ok[['price_actual']])

X_train = train_ok.loc[train_mask, feature_cols]
y_train = train_ok.loc[train_mask, 'price_actual_scaled']
X_val = train_ok.loc[val_mask, feature_cols]
y_val = train_ok.loc[val_mask, 'price_actual_scaled']
X_test = test_ok[feature_cols]

print(f"train rows: {len(X_train)}   val rows: {len(X_val)}")

X_train = X_train.drop(columns=['price_actual_scaled'])
X_val = X_val.drop(columns=['price_actual_scaled'])
X_test = X_test.drop(columns=['price_actual_scaled'])

X_train
y_train

X_val
y_val

X_test

np.isnan(X_test).sum(), np.isinf(X_test).sum()
np.isnan(X_train).sum(), np.isinf(X_train).sum()

print(f"test rows: {len(X_test)}")

# 前24時間分のデータを参照する
timesteps = 24

data_x = []
xarr = np.array
for i in range(timesteps, X_train.shape[0]):
    xset = []
    for j in range(X_train.shape[1]):
        d = X_train.iloc[i-timesteps:i, j]
        xset.append(d)
    xarr = np.array(xset).reshape(timesteps, X_train.shape[1])
    data_x.append(xarr)
X_train = np.array(data_x)

# 24件目以降のデータを目的変数に
y_train = y_train[timesteps:]

#valデータも同様に変換
data_x = []
xarr = np.array
for i in range(timesteps, X_val.shape[0]):
    xset = []
    for j in range(X_val.shape[1]):
        d = X_val.iloc[i-timesteps:i, j]
        xset.append(d)
    xarr = np.array(xset).reshape(timesteps, X_val.shape[1])
    data_x.append(xarr)
X_val = np.array(data_x)

# 24件目以降のデータを目的変数に
y_val = y_val[timesteps:]

#testデータも同様に変換
data_x = []
xarr = np.array
for i in range(timesteps, X_test.shape[0]):
    xset = []
    for j in range(X_test.shape[1]):
        d = X_test.iloc[i-timesteps:i, j]
        xset.append(d)
    xarr = np.array(xset).reshape(timesteps, X_test.shape[1])
    data_x.append(xarr)
X_test = np.array(data_x)

np.isnan(X_test).sum(), np.isinf(X_test).sum()


print(len(X_train))
print(len(X_val))

%%time

from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, GRU
from keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Input

neuron = 128
actfunc = "tanh"
dropout = 0.2
epochs = 300

model_LSTM_128 = Sequential()
# model_LSTM_128.add(LSTM(neuron, activation=actfunc, batch_input_shape=(None, timesteps, X_train.shape[2]), return_sequences=False))
model_LSTM_128.add(Input(shape=(timesteps, X_train.shape[2])))  # ←ここで shape を指定
model_LSTM_128.add(LSTM(neuron, activation=actfunc, return_sequences=False))
model_LSTM_128.add(Dropout(dropout))
model_LSTM_128.add(Dense(1, activation="linear"))

model_LSTM_128.compile(loss="mean_squared_error", optimizer="adam")
early_stopping =  EarlyStopping(monitor='val_loss', min_delta=0.0, patience=3)

history_LSTM_128 = model_LSTM_128.fit(X_train, y_train, batch_size=128, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])

plt.figure(figsize=(15,8))
plt.plot(history_LSTM_128.history['loss'], label='Train Loss')
plt.plot(history_LSTM_128.history['val_loss'], label='valid Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()
print(len(X_test))


from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


# 予測
y_val_pred_LSTM_128 = model_LSTM_128.predict(X_val)

# スケールを元に戻す
y_val_pred_LSTM_128 = scaler_y.inverse_transform(y_val_pred_LSTM_128)
y_t = scaler_y.inverse_transform(y_val.values.reshape(-1, 1))  # reshapeが必要！

# 指標計算
LSTM_RMSE_128 = np.sqrt(mean_squared_error(y_t, y_val_pred_LSTM_128))
LSTM_MAE_128 = mean_absolute_error(y_t, y_val_pred_LSTM_128)
LSTM_MAPE_128 = mean_absolute_percentage_error(y_t, y_val_pred_LSTM_128)

# 結果出力
print('RMSE:', LSTM_RMSE_128)
print('MAE:', LSTM_MAE_128)
print('MAPE:', LSTM_MAPE_128)

y_val_pred_LSTM_128

print(type(X_val))
print(X_val.shape)
print(X_train.shape)
print(type(X_test))
print(X_test.shape)          # (サンプル数, timesteps, 特徴量数) みたいになってるか？

X_test = X_test.astype('float32')

y_test_pred = model_LSTM_128.predict(X_test)

# スケールを元に戻す
y_test_pred = scaler_y.inverse_transform(y_test_pred)
y_p = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1))  # reshapeが必要！

#テストデータ(青)と予測(黄色)を描画
plt.figure(figsize=(30,8))
plt.plot(y_t, c="b")
plt.plot(y_val_pred_LSTM_128, c="y")
plt.show()

# 後ろから2週間分(24*7*2)のデータだけ描画
plt.figure(figsize=(30,8))
plt.plot(y_t[-336:], c="b")
plt.plot(y_val_pred_LSTM_128[-336:], c="y")
plt.show()

y_test_pred

# sample_submission に書き込み & 保存
# sample_submission_df[1] = y_test_pred

# 不足件数を計算
n_missing = len(sample_submission_df) - len(y_test_pred)

# 平均値を計算
avg = np.mean(y_test_pred)

# もし y_test_pred が shape = (8736, 1) のような 2次元配列なら、そのまま結合できるように
if n_missing > 0:
    pad = np.full((n_missing, 1), avg)  # 先頭に挿入する用の平均値配列
    y_test_pred = np.concatenate([pad, y_test_pred], axis=0)  # ← 順序を逆に

# 長さ確認
assert len(y_test_pred) == len(sample_submission_df)

# sample_submission に書き込み
sample_submission_df[1] = y_test_pred

sample_submission_df

sample_submission_df.to_csv(data_path+'LSTM256_ts24_0622.csv', header=False, index=False)
