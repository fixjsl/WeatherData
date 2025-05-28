import pandas as pd
import numpy as py
import os




station_section = \
{'강원도': ['속초', '철원', '대관령', '춘천', '강릉', '동해', '원주', '영월', '인제', '홍천', '태백'],
 '경기도': ['동두천', '수원', '양평', '이천'],
 '경상남도': ['창원', '통영', '진주', '거창', '합천', '밀양', '산청', '거제', '남해'],
 '경상북도': ['울진', '추풍령', '안동', '포항', '봉화', '영주', '문경', '영덕', '의성', '구미', '영천'],
 '광주광역시': ['광주'],
 '대구광역시': ['대구'],
 '대전광역시': ['대전'],
 '부산광역시': ['부산'],
 '서울특별시': ['서울'],
 '울산광역시': ['울산'],
 '인천광역시': ['백령도', '인천', '강화'],
 '전라남도': ['목포', '여수', '흑산도', '완도', '주암', '장흥', '해남', '고흥'],
 '전라북도': ['군산', '전주', '부안', '임실', '정읍', '남원', '장수'],
 '제주특별자치도': ['제주', '고산', '성산', '서귀포', '성산포'],
 '충청남도': ['서산', '천안', '보령', '부여', '금산'],
 '충청북도': ['충주', '청주', '제천', '보은']}


first_flag = {flag : True for flag in station_section.keys()}
os.makedirs("./WeatherDetail", exist_ok=True)
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
    for Key, stations in station_section.items():
        outpath = os.path.join("./WeatherDetail" ,f"{Key}.csv")
        
        for station in stations:
            tempdf = df[df["지점명"] == station]
            tempdf.to_csv(
                outpath,
                mode='w' if first_flag[Key] else 'a',
                header= first_flag[Key],
                index=False,
                encoding='utf-8')
            first_flag[Key] = False