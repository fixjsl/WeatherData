#연도별, 지역별 최후 개화일 그래프

import pandas as pd
import numpy as np
import seaborn as se
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
    filepath = "./WeatherDetail/4월미포함/"+지역+".csv"
    df = pd.read_csv(filepath)

    df["일시"] = pd.to_datetime(df["일시"])
    df["년도"] = df["일시"].dt.year
    df["월"] = df["일시"].dt.month

    filepath2 = "./개화일/"+지역+".csv"
    df2 = pd.read_csv(filepath2, encoding="utf-8")
    df2["벚나무"] = pd.to_datetime(df2["벚나무"])
    df2["년도"] = df2["벚나무"].dt.year

    mean = df.groupby("년도")["평균기온(°C)"].mean()

    #연최초개화일 = df.groupby("년도")["벚나무"].min()
    연최후개화일 = df2.groupby("년도")["벚나무"].max()

    #print(f"지역: {지역} ,연최초개화일", 연최초개화일)
    print(f"지역 : {지역} ,연최후개화일", 연최후개화일)
    print(mean)
    #print(f"지역: {지역} ,연최초개화일", 연최초개화일)
    plt.scatter( mean,연최후개화일.dt.dayofyear, label=지역)
plt.title("온도,개화일")
plt.xlabel("온도")
plt.ylabel("개화일")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()