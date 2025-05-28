import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import GRU
from sklearn.preprocessing import MinMaxScaler

지역들 = {
    "광주광역시": None, "대구광역시": None, "대전광역시": None,
    "서울특별시": None, "부산광역시": None, "울산광역시": None, "인천광역시": None
}

for key in 지역들.keys():

    filepath = f"./WeatherDetail/4월미포함/{key}.csv"
    filepath2 = f"./개화일/{key}.csv"

    df = pd.read_csv(filepath)
    df2 = pd.read_csv(filepath2)

    #날짜데이터를 datetime형식으로 변환
    df["일시"] = pd.to_datetime(df["일시"])
    df["년도"] = df["일시"].dt.year
    df["월"] = df["일시"].dt.month

    df2["벚나무"] = pd.to_datetime(df2["벚나무"])
    개화일 = df2["벚나무"].dt.dayofyear


    모든년도 = sorted(df["년도"].unique())

    #모든년도에 4월 인덱스를 추가
    full_index = pd.MultiIndex.from_product([모든년도, [2, 3, 4]], names=["년도", "월"])
    기온 = df.groupby(["년도", "월"])["평균기온(°C)"].sum().reindex(full_index).fillna(0)
    일조량 = df.groupby(["년도", "월"])["합계 일조시간(hr)"].sum().reindex(full_index).fillna(0)
    #월별 누적기온을 변환
    누적2월기온 = 기온.loc[pd.IndexSlice[:, 2]].to_numpy()
    누적3월기온 = 기온.loc[pd.IndexSlice[:, 3]].to_numpy()
    누적4월기온 = 기온.loc[pd.IndexSlice[:, 4]].to_numpy()
    #월별 누적 일조량을 넘파이로 변환
    누적2월일조 = 일조량.loc[pd.IndexSlice[:, 2]].to_numpy()
    누적3월일조 = 일조량.loc[pd.IndexSlice[:, 3]].to_numpy()
    누적4월일조 = 일조량.loc[pd.IndexSlice[:, 4]].to_numpy()

    #필요한 요소들을 column stack으로 쌓음
    X = np.column_stack([
        누적2월기온, 누적3월기온, 누적4월기온,
        누적2월일조, 누적3월일조, 누적4월일조
    ])
    y = 개화일.to_numpy()

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    X_train, X_test = X_scaled[:24], X_scaled[24:]
    y_train, y_test = y_scaled[:24], y_scaled[24:]
    #모델 생성
    model = Sequential()
    #GRU모델 에서 활성화 함수는 ReLu로
    model.add(GRU(50, activation='relu', input_shape=(1, 6)))
    #출력층은 하나
    model.add(Dense(1))
    #경사하강법은 adam사용 손실함수는 mse
    model.compile(optimizer='adam', loss='mse')
    #200에폭만큼 학습
    model.fit(X_train, y_train, epochs=200, verbose=0)

    #모델 테스트
    y_pred = model.predict(X_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred)
    y_test_inv = scaler_y.inverse_transform(y_test)

    print(f"지역 {key} 예측 개화일:", round(y_pred_inv[0][0]))
    print(f"지역 {key} 실제 개화일:", round(y_test_inv[0][0]))
    print("-" * 50)