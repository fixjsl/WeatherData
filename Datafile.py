#원본 자료를 지역별로 나누는 코드

import pandas as pd
import numpy as py
import os




station_section = \
{'광주광역시': ['광주'],
 '대구광역시': ['대구'],
 '대전광역시': ['대전'],
 '부산광역시': ['부산'],
 '서울특별시': ['서울'],
 '울산광역시': ['울산'],
 '인천광역시': ['백령도', '인천', '강화']}


first_flag = {flag : True for flag in station_section.keys()}
os.makedirs("./WeatherDetail", exist_ok=True)
for number in range(1,26):
    #현재 연도의 파일 가져오기
    year = 2000 + number
    filename = f"{year}.csv"
    filepath = "./원본 자료/" + filename
    df = pd.read_csv(filepath, encoding="euc-kr")

    for Key, stations in station_section.items():
        outpath = os.path.join("./원본 자료" ,f"{Key}.csv")
        
        for station in stations:
            tempdf = df[df["지점명"] == station]
            tempdf.to_csv(
                outpath,
                mode='w' if first_flag[Key] else 'a',
                header= first_flag[Key],
                index=False,
                encoding='utf-8')
            first_flag[Key] = False