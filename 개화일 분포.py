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
    filepath = "./개화일/"+지역+".csv"
    df = pd.read_csv(filepath, encoding="utf-8")
    df["벚나무"] = pd.to_datetime(df["벚나무"])
    df["년도"] = df["벚나무"].dt.year

    

    
    연최초개화일 = df.groupby("년도")["벚나무"].min()
    연최후개화일 = df.groupby("년도")["벚나무"].max()

    #print(f"지역: {지역} ,연최초개화일", 연최초개화일)
    
    print(f"지역 : {지역} ,연최후개화일", 연최초개화일)
    print 
    plt.plot(연최초개화일[:24].index, 연최초개화일.dt.dayofyear[:24], label=지역)
plt.title("연도별 개화일 (벚나무)")
plt.xlabel("년도")
plt.ylabel("개화일 (연중 일수 기준)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
