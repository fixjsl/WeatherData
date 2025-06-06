#RandomForest모델
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

#RandomForestModel 지역별로 
서울 = RandomForestRegressor(random_state=42)
대구 = RandomForestRegressor(random_state=42)
광주 = RandomForestRegressor(random_state=42)
대전 = RandomForestRegressor(random_state=42)
부산 = RandomForestRegressor(random_state=42)
인천 = RandomForestRegressor(random_state=42)
울산 = RandomForestRegressor(random_state=42)

지역들 = {
    "광주광역시": 광주, "대구광역시": 대구, "대전광역시": 대전,
    "서울특별시": 서울, "부산광역시": 부산, "울산광역시": 울산, "인천광역시": 인천
}
modeloutpath = "./Model/"
for key, station in 지역들.items():
    #지역별 자료 불러오기
    filepath = "./WeatherDetail/" + key + ".csv"
    filepath2 = "./개화일/" + key + ".csv"

    df = pd.read_csv(filepath)
    df2 = pd.read_csv(filepath2)

    os.makedirs("./Model/"+key, exist_ok=True)
    modeloupath = os.path.join(modeloutpath,key,"RandomForest.pkl")
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

    #날짜데이터를 datetime형식으로 변환
    df2["벚나무"] = pd.to_datetime(df2["벚나무"])
    개화일 = df2["벚나무"].dt.dayofyear
    개화일np = 개화일.loc[:19].to_numpy()  # 학습용 정답
    T개화일np = 개화일.loc[20:23] #테스트용 정답
    Training_data = Train_data[:20]  # 마지막 1년 제외
    Test_data = Train_data[20:24] # 테스트용 데이터
    

    
    station.fit(Training_data, 개화일np)

    
    forecast  = station.predict([Test_data[0]])[0]
    forecast2 = station.predict([Test_data[1]])[0]
    forecast3 = station.predict([Test_data[2]])[0]
    forecast4 = station.predict([Test_data[3]])[0]

    print(f"지역 : {key}, 정확도(R²): ", station.score(Training_data, 개화일np))
    print(f"지역 : {key}, 예측값(R²): ", forecast)
    print(f"지역 : {key}, 예측값(R²): ", forecast2)
    print(f"지역 : {key}, 예측값(R²): ", forecast3)
    print(f"지역 : {key}, 예측값(R²): ", forecast4)
    print(f"지역 : {key}, 실제값(R²):\n", T개화일np)
    
    joblib.dump(station,modeloupath)


