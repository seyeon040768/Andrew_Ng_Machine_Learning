# Coursera Lectures
- Week1
  - [Introduction](#introduction)
    - [1. What is Machine Learning?](#1-what-is-machine-learning) 
    - [2. Supervised Learning](#2-supervised-learning)
    - [3. Unsupervised Learning](#3-unsupervised-learning)
  - [Model and Cost Function](#model-and-cost-function)
    - [1. Model Representation](#1-model-representation)

## Introduction
### 1. What is Machine Learning?
컴퓨터가 명시적으로 프로그램되지 않고도 학습할 수 있도록 하는 연구 분야 by Arthur Samuel

만약 어떤 작업 T에서 경험 E를 통해 성능 측정 방법인 P로 측정했을 때 성능이 향상된다면 이런 컴퓨터 프로그램은 학습을 한다고 말한다. by Tom Mitchell
***
### 2. Supervised Learning
이미 결과가 알려진 데이터를 주고 학습시키는 것

**Regresstion(회귀)**   
주어진 데이터를 바탕으로 값을 예측하는 것   
ex) 평수로 주택 가격 예측   
<img src="./Week1/Normdist_regression.png" width="30%">

**Classification(분류)**   
어떤 데이터를 여러 값중 하나로 분류하는 것   
ex) 종양 크기로 악성 종양인지 판단   
<img src="./Week1/classification.png" width="50%">
***
### 3. Unsupervised Learning
정답을 알려주지 않고 데이터를 군집화 하는것   
데이터가 무엇인지는 정의할 수 없지만 비슷한 특징을 찾아 분류   
ex) 뉴스 기사 분류
***
## Model and Cost Function
### 1. Model Representation
<img src="./Week1/model representation.PNG" width="50%">

m : 데이터의 총 개수   
x^(i) : i번째 x 데이터  ex) x^(2) : 1416   
y^(i) : i번째 y 데이터  ex) y^(1) : 460

Supervised Learning(지도 학습)의 목표는 y = h(x)에서 좋은 y값을 예측하는 것이 목표   
여기서 h를 hypothesis(가설)이라고 함
<img src="./Week1/hypothesis.PNG" width="50%">

<img src="./Week1/process.png" width="50%">

***

