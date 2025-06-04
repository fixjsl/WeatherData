#Prophet모델
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor
from prophet import Prophet
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import joblib as jo
import os


지역들 = {
    "광주광역시": Prophet(yearly_seasonality=True),
    "대구광역시": Prophet(yearly_seasonality=True),
    "대전광역시": Prophet(yearly_seasonality=True),
    "서울특별시": Prophet(yearly_seasonality=True),
    "부산광역시": Prophet(yearly_seasonality=True),
    "울산광역시": Prophet(yearly_seasonality=True),
    "인천광역시": Prophet(yearly_seasonality=True),
}


modeloutpath = "./Model/"
for key, station in 지역들.items():

    filepath = "./WeatherDetail/"+key+".csv"

    filepath2 = "./개화일/"+key+".csv"

    df = pd.read_csv(filepath)
    df2 = pd.read_csv(filepath2)
    modeloupath = os.path.join(modeloutpath,key,"Prophet.pkl")
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


    Train_data = np.column_stack([누적2월기온,누적3월기온,누적4월기온,누적2월일조량,누적3월일조량,누적4월일조량])

    개화일 = df2["벚나무"].dt.dayofyear
    

    
    개화일np = 개화일.loc[:19].to_numpy()  # 학습용 정답
    T개화일np = 개화일.loc[20:23] #테스트용 정답
    Training_data = Train_data[:20]  # 마지막 1년 제외
    Test_data = Train_data[20:24] # 테스트용 데이터

    # pandas로 변환
    train_df = pd.DataFrame(Training_data, columns=[
    "2월기온", "3월기온", "4월기온",
    "2월일조", "3월일조", "4월일조"
    ]   )
    train_df["y"] = 개화일np
# 연도 정보: 그냥 1월 1일로 세팅
    years = df["년도"].unique()[:20]  # 24년치
    train_df["ds"] = pd.to_datetime(np.char.add(years.astype(str), "-01-01"))

# Prophet 모델 생성 및 외부 변수 등록
    columns =  ["2월기온", "3월기온", "4월기온", "2월일조", "3월일조", "4월일조"]
    for col in columns:
        station.add_regressor(col)

# 학습
    station.fit(train_df[["ds", "y"] + [col for col in train_df.columns if col.startswith(("2월", "3월", "4월"))]])

# 예측 데이터셋 (예: 마지막 연도 1건)
    years_test = df["년도"].unique()[21:25]
    test_df = pd.DataFrame(Test_data, columns=columns).assign(
    ds=pd.to_datetime(years_test.astype(str) + "-01-01")
)

#점수
    

    

    


# 예측
    forecast = station.predict(test_df)
    예측값 = forecast["yhat"].values
    실제값 = T개화일np
    점수 = r2_score(실제값,예측값)
    print(f"지역 : {key} ,점수 : " , 점수)
    print(f"지역 : {key} ,예측 개화일 (20):", forecast["yhat"].values[0])
    print(f"지역 : {key} ,예측 개화일 (21):", forecast["yhat"].values[1])
    print(f"지역 : {key} ,예측 개화일 (22):", forecast["yhat"].values[2])
    print(f"지역 : {key} ,예측 개화일 (23):", forecast["yhat"].values[3])
    print(f"지역 : {key} ,실제 개화일:", T개화일np)
    jo.dump(station,modeloupath)