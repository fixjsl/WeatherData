import pandas as pd
import numpy as np
import joblib as jo
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
import matplotlib.pyplot as plt
import os






지역들 = ["광주광역시", "대구광역시", "대전광역시",
    "서울특별시", "부산광역시", "울산광역시", "인천광역시"]
modeloutpath = "./Model/"
for key in 지역들:
    #지역별 자료 불러오기
    filepath = "./WeatherDetail/" + key + ".csv"
    filepath2 = "./개화일/" + key + ".csv"

    df = pd.read_csv(filepath)
    df2 = pd.read_csv(filepath2)


    os.makedirs("./Model/"+key, exist_ok=True)
    Randommodelpath = os.path.join(modeloutpath,key,"RandomForest.pkl")
    LinerModelpath =os.path.join(modeloutpath,key,"Linear.pkl")
    ProphetModelpath = os.path.join(modeloutpath,key,"Prophet.pkl")
    Random  : RandomForestRegressor= jo.load(Randommodelpath)
    Linear : LinearRegression = jo.load(LinerModelpath)
    Prop : Prophet = jo.load(ProphetModelpath)
    #날짜데이터를 datetime형식으로 변환
    df["일시"] = pd.to_datetime(df["일시"])
    df["년도"] = df["일시"].dt.year
    df["월"] = df["일시"].dt.month

    모든년도 = df["년도"].unique()
    # MultiIndex 생성 (2,3,4월 모두)
    full_index = pd.MultiIndex.from_product(
        [sorted(모든년도), [2, 3, 4]],
        names=["년도", "월"]
    )
    # 연도별 누적 기온
    연누적기온_pd = df.groupby(["년도", "월"])["평균기온(°C)"].sum().reindex(full_index).fillna(0)
    연누적일조량_pd = df.groupby(["년도", "월"])["합계 일조시간(hr)"].sum().reindex(full_index).fillna(0)
    #월별 누적기온으로 분류
    누적2월기온 = 연누적기온_pd.loc[pd.IndexSlice[:, 2]].to_numpy()
    누적3월기온 = 연누적기온_pd.loc[pd.IndexSlice[:, 3]].to_numpy()
    누적4월기온 = 연누적기온_pd.loc[pd.IndexSlice[:, 4]].to_numpy()

    누적2월일조량 = 연누적일조량_pd.loc[pd.IndexSlice[:, 2]].to_numpy()
    누적3월일조량 = 연누적일조량_pd.loc[pd.IndexSlice[:, 3]].to_numpy()
    누적4월일조량 = 연누적일조량_pd.loc[pd.IndexSlice[:, 4]].to_numpy()
    #학습데이터 만들기
    Train_data = np.column_stack([
        누적2월기온, 누적3월기온, 누적4월기온,
        누적2월일조량, 누적3월일조량, 누적4월일조량
    ])
    columns =  ["2월기온", "3월기온", "4월기온", "2월일조", "3월일조", "4월일조"]
    #날짜데이터를 datetime형식으로 변환
    df2["벚나무"] = pd.to_datetime(df2["벚나무"])
    개화일 = df2["벚나무"].dt.dayofyear
    T개화일np = 개화일.loc[24:] #테스트용 정답

    Test_data = Train_data[24:] # 테스트용 데이터

    years_test = df["년도"].unique()[24:]
    test_df = pd.DataFrame(Test_data, columns=columns).assign(
    ds=pd.to_datetime(years_test.astype(str) + "-01-01")
)

    forecastLinear = Linear.predict(Test_data)
    forecastRandom = Random.predict(Test_data)
    forecastProp = Prop.predict(test_df)
    print(f"지역 : {key}, Linear예측값(R²): ", forecastLinear)
    print(f"지역 : {key}, Random예측값(R²): ", forecastRandom)
    print(f"지역 : {key}, Prophet예측값(R²): ", forecastProp["yhat"].values[0])
    print(f"지역 : {key}, 실제값(R²):\n", T개화일np)
    