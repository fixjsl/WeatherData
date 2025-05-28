import pandas as pd
import numpy as np
import seaborn as se
import matplotlib.pyplot as plt
import os


지역들 = ["광주광역시","대구광역시","대전광역시","서울특별시","부산광역시","울산광역시","인천광역시"]



for 지역 in 지역들:

    filepath = "./WeatherDetail/4월포함/"+지역+".csv"

    filepath2 = "./개화일/"+지역+".csv"

         

    df = pd.read_csv(filepath)
    df2 = pd.read_csv(filepath2)


    #날짜데이터를 datetime형식으로 변환
    df2["벚나무"] = pd.to_datetime(df2["벚나무"])
    df["일시"] = pd.to_datetime(df["일시"])
    #벚꽃이 피어난 날
    max = df2["벚나무"].dt.dayofyear.max()

    
    #벚꽃이 피어난 날 까지만 등록 후 csv로 내보내기
    df =  df[df["일시"].dt.dayofyear <= max] 
    df.to_csv("./WeatherDetail/4월미포함/"+지역+".csv", index= False , encoding= 'utf-8')
