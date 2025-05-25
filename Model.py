import pandas as pd
import numpy as np
import seaborn as se
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor
from prophet import Prophet
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os


서울 = Prophet(yearly_seasonality=True)#LinearRegression()
대구 = Prophet(yearly_seasonality=True)#LinearRegression()
광주 = Prophet(yearly_seasonality=True)#LinearRegression()
대전 = Prophet(yearly_seasonality=True)#LinearRegression()
부산 = Prophet(yearly_seasonality=True)#LinearRegression()
인천 = Prophet(yearly_seasonality=True)#LinearRegression()
울산 = Prophet(yearly_seasonality=True)#LinearRegression()

지역들 = {"광주광역시":광주,"대구광역시":대구,"대전광역시":대전,"서울특별시":서울,"부산광역시":부산,"울산광역시":울산,"인천광역시":인천}


for key, station in 지역들.items():

    filepath = "./WeatherDetail/4월미포함/"+key+".csv"

    filepath2 = "./개화일/"+key+".csv"

    df = pd.read_csv(filepath)
    df2 = pd.read_csv(filepath2)

    df["일시"] = pd.to_datetime(df["일시"])
    df["년도"] = df["일시"].dt.year
    df["월"] = df["일시"].dt.month

    df2["벚나무"] = pd.to_datetime(df2["벚나무"])


    모든년도 = df["년도"].unique()

    # MultiIndex 생성 (2,3,4월 모두)
    full_index = pd.MultiIndex.from_product(\
    [sorted(모든년도), [2, 3, 4]],\
    names=["년도", "월"]\
    )
    # 연도별 평균 기온
    연누적기온_pd = df.groupby(["년도","월"])["평균기온(°C)"].sum().reindex(full_index).fillna(0)
    연누적일조량_pd = df.groupby(["년도","월"])["합계 일조시간(hr)"].sum().reindex(full_index).fillna(0) 


    #print(연누적기온_pd)

    누적2월기온 = 연누적기온_pd.loc[pd.IndexSlice[:,2]].to_numpy()
    누적3월기온 = 연누적기온_pd.loc[pd.IndexSlice[:,3]].to_numpy()
    누적4월기온 = 연누적기온_pd.loc[pd.IndexSlice[:,4]].to_numpy()

    누적2월일조량 = 연누적일조량_pd.loc[pd.IndexSlice[:,2]].to_numpy()
    누적3월일조량 = 연누적일조량_pd.loc[pd.IndexSlice[:,3]].to_numpy()
    누적4월일조량 = 연누적일조량_pd.loc[pd.IndexSlice[:,4]].to_numpy()


    Train_data = np.column_stack([누적2월기온,누적3월기온,누적4월기온,누적2월일조량,누적3월일조량,누적4월일조량])
    #print(Train_data)




    #print(연누적기온_pd.index.size)

    개화일 = df2["벚나무"].dt.dayofyear
    

    연누적기온np = 연누적기온_pd[:연누적기온_pd.index.size-3].to_numpy()
    연누적일조량np = 연누적일조량_pd[:연누적기온_pd.index.size-3].to_numpy()


    T연누적기온np = 연누적기온_pd.tail(3).to_numpy()
    T연누적일조량np = 연누적일조량_pd.tail(3).to_numpy()

    
    개화일np = 개화일.loc[:23].to_numpy()
    T개화일np = 개화일.loc[24]


    Training_data = Train_data[:24]
    Test_data = Train_data[24]



    #print(f"지역 데이터(학습) {key} \n",Training_data)
    #print(f"지역 : {key} 정답 개화일 \n{개화일np}")

    #print(f"지역 데이터(정답) {key}\n",Test_data)
    #print(f"지역 : {key} 정답 개화일 \n{T개화일np}")

    #station.fit(Training_data, 개화일np)

    
    #print(f"지역 : {key}, 점수 : ",station.score(Training_data,개화일np))

    #예측값 = station.predict([Test_data])
    #print("예측 개화일:", 예측값)
    #print("실제 개화일:", T개화일np)

    # pandas로 변환
    train_df = pd.DataFrame(Training_data, columns=[
    "2월기온", "3월기온", "4월기온",
    "2월일조", "3월일조", "4월일조"
    ]   )
    train_df["y"] = 개화일np

# 연도 정보: 그냥 1월 1일로 세팅
    years = df["년도"].unique()[:24]  # 24년치
    train_df["ds"] = pd.to_datetime(years.astype(str) + "-01-01")

# Prophet 모델 생성 및 외부 변수 등록
    
    for col in ["2월기온", "3월기온", "4월기온", "2월일조", "3월일조", "4월일조"]:
        station.add_regressor(col)

# 학습
    station.fit(train_df[["ds", "y"] + [col for col in train_df.columns if col.startswith(("2월", "3월", "4월"))]])

# 예측 데이터셋 (예: 마지막 연도 1건)
    test_df = pd.DataFrame([Test_data], columns=[
    "2월기온", "3월기온", "4월기온",
    "2월일조", "3월일조", "4월일조"
    ])
    test_df["ds"] = pd.to_datetime(["2025-01-01"])

#점수
    train_forecast = station.predict(train_df)
    예측값 = train_forecast["yhat"].values
    실제값 = train_df["y"].values

    점수 = r2_score(실제값,예측값)

    print(f"지역 : {key} ,점수 : " , 점수)


# 예측
    forecast = station.predict(test_df)


    
    print(f"지역 : {key} ,예측 개화일 (yhat):", forecast["yhat"].values[0])
    print(f"지역 : {key} ,실제 개화일:", T개화일np)
