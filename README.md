# ATAR: Automatic and Tunable Artifact Removal Algorithm

시계열 data(eeg data)를 pandas와 signal processing toolkit을 이용하여 전처리하고, Artifact를 Remove를 하여 뇌파 결과를 분석

## 실행과정

- unity에서 측정한 뇌파에서 잘못된 이름값을 전처리
- 뇌파를 시간에 따라서 cutting하고 값을 저장
- atar algorithm을 이용해서 뇌파가아닌 잡음(걸었을때, 눈을 움직일 때 생기는 신호 etc...)을 remove
- result 확인

## Block Diagram

![피그마](img/Block.PNG)
