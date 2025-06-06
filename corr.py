#상관관계 계산 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

지역들 = ["광주광역시","대구광역시","대전광역시","서울특별시","부산광역시","울산광역시","인천광역시"]


for 지역 in 지역들:

    filepath = "./WeatherDetail/"+지역+".csv"

    filepath2 = "./개화일/"+지역+".csv"

    df = pd.read_csv(filepath)
    df2 = pd.read_csv(filepath2)

    #날짜데이터를 datetime형식으로 변환
    df["일시"] = pd.to_datetime(df["일시"])
    df["년도"] = df["일시"].dt.year
    df["월"] = df["일시"].dt.month

    # 연도별 누적 기온
    연누적기온 = df.groupby(["년도","월"])["평균기온(°C)"].sum()
    연평균2 = df.groupby(["년도","월"])["최저기온(°C)"].mean()
    연평균3 = df.groupby(["년도","월"])["최고기온(°C)"].mean()
    연누적일조량 = df.groupby(["년도","월"])["합계 일조시간(hr)"].sum()
    연누적강수량 = df.groupby(["년도","월"])["일강수량(mm)"].sum()

    #print(f"연누적기온  : {연누적기온}")
    # print(f"연최저기온평균  : {연평균2}")
    # print(f"연최고기운평균  : {연평균3}")
    #print(f"연누적일조량  : {연누적일조량}")
    # print(f"연누적강수량  : {연누적강수량}")

    # 개화일 → 연도 + 연중 몇 번째 날인지
    개화일 = pd.to_datetime(df2["벚나무"])
    df2["년도"] = 개화일.dt.year
    df2["월"] = 개화일.dt.month
    df2["개화일숫자"] = 개화일.dt.dayofyear

    # 인덱스 맞춰서 연도별, 월 기준으로 정렬
    개화일숫자 = df2.set_index(["년도","월"])["개화일숫자"]

    # 인덱스를 연도 기준으로 맞춰야 corr 가능
    공통연도 = 연누적기온.index.intersection(개화일숫자.index)
    상관관계 = 연누적기온.loc[공통연도].corr(개화일숫자.loc[공통연도])

    공통연도 = 연평균2.index.intersection(개화일숫자.index)
    상관관계2 = 연평균2.loc[공통연도].corr(개화일숫자.loc[공통연도])
    공통연도 = 연평균3.index.intersection(개화일숫자.index)
    상관관계3 = 연평균3.loc[공통연도].corr(개화일숫자.loc[공통연도])
    공통연도 = 연누적일조량.index.intersection(개화일숫자.index)
    상관관계4 = 연누적일조량.loc[공통연도].corr(개화일숫자.loc[공통연도])
    공통연도 = 연누적강수량.index.intersection(개화일숫자.index)
    상관관계5 = 연누적강수량.loc[공통연도].corr(개화일숫자.loc[공통연도])

    print(지역)
    print("개화일 vs 연누적기온 상관관계:", 상관관계)
    print("개화일 vs 연최저기온평균 상관관계:", 상관관계2)

    print("개화일 vs 연최고기온평균균 상관관계:", 상관관계3)


    print("개화일 vs 연누적일조량 상관관계:", 상관관계4)

    print("개화일 vs 연누적강수량 상관관계:", 상관관계5)

