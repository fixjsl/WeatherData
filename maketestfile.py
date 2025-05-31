import pandas as pd
import numpy as py
import os


testStation = ["수원", "흑산도", "전주","청주","안동","목포","여수","주암","창원","포항"]

first_flag = {flag : True for flag in testStation}
os.makedirs("./TestData/problem", exist_ok=True)
for number in range(1,26):
    #현재 연도의 파일 가져오기
    year = 2000 + number
    filename = f"{year}.csv"
    filepath = "./원본 자료/" + filename
    df = pd.read_csv(filepath, encoding="euc-kr")
    df = df.fillna({
    "일 최심적설(cm)": 0,
    "평균 전운량(1/10)": 0
    })
    
        
    for station in testStation:
        outpath = os.path.join("./TestData/problem" ,f"{station}.csv")
        tempdf = df[df["지점명"] == station]
        tempdf.to_csv(
            outpath,
                mode='w' if first_flag[station] else 'a',
                header= first_flag[station],
                index=False,
                encoding='utf-8')
        first_flag[station] = False