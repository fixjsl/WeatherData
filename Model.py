import pandas as pd
import numpy as np
import seaborn as se
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor
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


for key, station in 지역들.items():

    filepath = "./WeatherDetail/4월미포함/"+key+".csv"

    filepath2 = "./개화일/"+key+".csv"

    df = pd.read_csv(filepath)
    df2 = pd.read_csv(filepath2)

    df["일시"] = pd.to_datetime(df["일시"])
    df["년도"] = df["일시"].dt.year
    df["월"] = df["일시"].dt.month

    # 연도별 평균 기온
    연누적기온_pd = df.groupby(["년도","월"])["평균기온(°C)"].sum()
    연누적일조량_pd = df.groupby(["년도","월"])["합계 일조시간(hr)"].sum() 

    연누적기온np = 연누적기온_pd.to_numpy()
    연누적일조량np = 연누적일조량_pd.to_numpy()

    data = np.concatenate([연누적기온np,연누적일조량np])
    print(data)
