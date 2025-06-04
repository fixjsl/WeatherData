#선형회귀 모델 코드
import pandas as pd
import numpy as np
import joblib as jo
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os


서울 = LinearRegression()
대구 = LinearRegression()
광주 = LinearRegression()
대전 = LinearRegression()
부산 = LinearRegression()
인천 = LinearRegression()
울산 = LinearRegression()

지역들 = {"광주광역시":광주,"대구광역시":대구,"대전광역시":대전,"서울특별시":서울,"부산광역시":부산,"울산광역시":울산,"인천광역시":인천}

modeloutpath = "./Model/"
for key, station in 지역들.items():

    filepath = "./WeatherDetail/"+key+".csv"

    filepath2 = "./개화일/"+key+".csv"

    df = pd.read_csv(filepath)
    df2 = pd.read_csv(filepath2)
    os.makedirs("./Model/"+key, exist_ok=True)
    modeloupath = os.path.join(modeloutpath,key,"Linear.pkl")
    #날짜데이터를 datetime형식으로 변환
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

    누적2월기온 = 연누적기온_pd.loc[pd.IndexSlice[:,2]].to_numpy()
    누적3월기온 = 연누적기온_pd.loc[pd.IndexSlice[:,3]].to_numpy()
    누적4월기온 = 연누적기온_pd.loc[pd.IndexSlice[:,4]].to_numpy()

    누적2월일조량 = 연누적일조량_pd.loc[pd.IndexSlice[:,2]].to_numpy()
    누적3월일조량 = 연누적일조량_pd.loc[pd.IndexSlice[:,3]].to_numpy()
    누적4월일조량 = 연누적일조량_pd.loc[pd.IndexSlice[:,4]].to_numpy()

    #학습데이터 배열 변형
    Train_data = np.column_stack([누적2월기온,누적3월기온,누적4월기온,누적2월일조량,누적3월일조량,누적4월일조량])

    개화일 = df2["벚나무"].dt.dayofyear
    
    연누적기온np = 연누적기온_pd[:연누적기온_pd.index.size-3].to_numpy()
    연누적일조량np = 연누적일조량_pd[:연누적기온_pd.index.size-3].to_numpy()

    T연누적기온np = 연누적기온_pd.tail(3).to_numpy()
    T연누적일조량np = 연누적일조량_pd.tail(3).to_numpy()

    개화일np = 개화일.loc[:19].to_numpy()  # 학습용 정답
    T개화일np = 개화일.loc[20:23] #테스트용 정답
    Training_data = Train_data[:20]  # 마지막 1년 제외
    Test_data = Train_data[20:24] # 테스트용 데이터
        
    station.fit(Training_data, 개화일np)

    forecast  = station.predict([Test_data[0]])[0]
    forecast2 = station.predict([Test_data[1]])[0]
    forecast3 = station.predict([Test_data[2]])[0]
    forecast4 = station.predict([Test_data[3]])[0]

    print(f"지역 : {key}, 정확도(R²): ", station.score(Test_data, T개화일np))
    print(f"지역 : {key}, 예측값(R²): ", forecast)
    print(f"지역 : {key}, 예측값(R²): ", forecast2)
    print(f"지역 : {key}, 예측값(R²): ", forecast3)
    print(f"지역 : {key}, 예측값(R²): ", forecast4)
    print(f"지역 : {key}, 실제값(R²):\n", T개화일np)

    jo.dump(station, modeloupath)