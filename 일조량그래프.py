#연도별, 지역별 최후 개화일 그래프

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 윈도우에 설치된 기본 한글 폰트 경로 (예: 맑은 고딕)
font_path = "C:/Windows/Fonts/malgun.ttf"  # 또는 NanumGothic, batang 등

# 폰트 등록
fontprop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()

#
지역들 = ["광주광역시","대구광역시","대전광역시","서울특별시","부산광역시","울산광역시","인천광역시"]

for 지역 in 지역들:
    filepath = "./WeatherDetail/"+지역+".csv"
    df = pd.read_csv(filepath)

    df["일시"] = pd.to_datetime(df["일시"])
    df["년도"] = df["일시"].dt.year
    df["월"] = df["일시"].dt.month

    filepath2 = "./개화일/"+지역+".csv"
    df2 = pd.read_csv(filepath2, encoding="utf-8")
    df2["벚나무"] = pd.to_datetime(df2["벚나무"])
    df2["년도"] = df2["벚나무"].dt.year

    mean = df.groupby("년도")["합계 일조시간(hr)"].sum()

    #연최초개화일 = df.groupby("년도")["벚나무"].min()
    연최후개화일 = df2.groupby("년도")["벚나무"].max()
    x = mean[:24]
    y = 연최후개화일[:24].dt.dayofyear
    slope, intercept = np.polyfit(x, y, 1)

    y2 = slope * mean + intercept

    #print(f"지역: {지역} ,연최초개화일", 연최초개화일)
    print(f"지역 : {지역} ,연최후개화일", 연최후개화일)
    print(mean)
    #print(f"지역: {지역} ,연최초개화일", 연최초개화일)
    plt.scatter( mean[:24],연최후개화일[:24].dt.dayofyear, label=지역)
    plt.plot(mean,y2)
    plt.title("일조량,개화일")
    plt.xlabel("일조량")
    plt.ylabel("개화일")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()