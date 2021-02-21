# Coursera Lectures
- [Week1](#week1)
  - [Introduction](#introduction)
    - [1. What is Machine Learning?](#1-what-is-machine-learning) 
    - [2. Supervised Learning](#2-supervised-learning)
    - [3. Unsupervised Learning](#3-unsupervised-learning)
  - [Model and Cost Function](#model-and-cost-function)
    - [1. Model Representation](#1-model-representation)
    - [2. Cost Function](#2-cost-function)
    - [3. Cost Function Intuition 1](#3-cost-function-intuition-1
# Week1
## Introduction
### 1. What is Machine Learning?
컴퓨터가 명시적으로 프로그램되지 않고도 학습할 수 있도록 하는 연구 분야 by Arthur Samuel

만약 어떤 작업 T에서 경험 E를 통해 성능 측정 방법인 P로 측정했을 때 성능이 향상된다면 이런 컴퓨터 프로그램은 학습을 한다고 말한다. by Tom Mitchell
***
### 2. Supervised Learning
**지도 학습**   
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
**비지도 학습**   
정답을 알려주지 않고 데이터를 군집화 하는것   
데이터가 무엇인지는 정의할 수 없지만 비슷한 특징을 찾아 분류   
ex) 뉴스 기사 분류
***
## Model and Cost Function
### 1. Model Representation
<img src="./Week1/model representation.PNG" width="50%">

<img src="https://latex.codecogs.com/gif.latex?m" /> : 데이터의 총 개수   
<img src="https://latex.codecogs.com/gif.latex?x^{(i)}" /> : <img src="https://latex.codecogs.com/gif.latex?i" />번째 <img src="https://latex.codecogs.com/gif.latex?x" /> 데이터  ex) <img src="https://latex.codecogs.com/gif.latex?x^{(2)}" /> : <img src="https://latex.codecogs.com/gif.latex?1416" />   
<img src="https://latex.codecogs.com/gif.latex?y^{(i)}" /> : <img src="https://latex.codecogs.com/gif.latex?i" />번째 <img src="https://latex.codecogs.com/gif.latex?y" /> 데이터  ex) <img src="https://latex.codecogs.com/gif.latex?y^{(1)}" /> : <img src="https://latex.codecogs.com/gif.latex?460" />

Supervised Learning(지도 학습)의 목표는 <img src="https://latex.codecogs.com/gif.latex?h(x)" />를 <img src="https://latex.codecogs.com/gif.latex?y" />값에 가깝게 만드는 것이 목표   
여기서 <img src="https://latex.codecogs.com/gif.latex?h" />를 hypothesis(가설)이라고 함   

<img src="./Week1/hypothesis.PNG" width="50%">

<img src="./Week1/process.png" width="50%">

***
### 2. Cost Function
설정한 가설( <img src="https://latex.codecogs.com/gif.latex?h(x)" /> )의 정확도를 확인하기 위해 Cost Function(비용 함수)를 사용   
비용 함수의 값이 작을수록(0에 가까울수록) 정확   

비용 함수는 아래와 같이 **Squared error function** or **Mean squared error**(평균 제곱 오차) 방식을 주로 씀
<img src="./Week1/cost_function.png" width="50%">

***
### 3. Cost Function Intuition 1
<img src="https://latex.codecogs.com/gif.latex?h(x)=\Theta_{0}+\Theta_{1}x" />에서 <img src="https://latex.codecogs.com/gif.latex?\Theta_{0}=0" />이라고 가정

데이터 셋:   
<img src="https://latex.codecogs.com/gif.latex?(1,1)" />   
<img src="https://latex.codecogs.com/gif.latex?(2,2)" />   
<img src="https://latex.codecogs.com/gif.latex?(3,3)" />   

<img src="./Week1/cost_function_1.png" width="50%">

<img src="https://latex.codecogs.com/gif.latex?\Theta_{1}=1" />이면 위 그래프와 같이 데이터와 완벽이 일치한다. 이때 비용 함수를 구해 보면   
<img src="https://latex.codecogs.com/gif.latex?\frac{1}{2\times3}\sum_{i=1}^{3}(h_{\Theta}(x_{i})-y_{i})^{2}=\frac{1}{6}(0^{2}+0^{2}+0^{2})=0" />   
과 같이 <img src="https://latex.codecogs.com/gif.latex?0" />이 나온다.

<img src="./Week1/cost_function_2.png" width="50%">

<img src="https://latex.codecogs.com/gif.latex?\Theta_{1}=0.5" />이면 위 그래프와 같은 모양이 나온다. 여기서 비용 함수를 구해 보면   
<img src="https://latex.codecogs.com/gif.latex?\frac{1}{2\times3}\sum_{i=1}^{3}(h_{\Theta}(x_{i})-y_{i})^{2}=\frac{1}{6}(0.5^{2}+1^{2}+1.5^{2})=\frac{1}{6}\times\frac{7}{2}\simeq0.58" />과 같이 약 <img src="https://latex.codecogs.com/gif.latex?0.58" />이 나온다. 아까 데이터와 그래프가 완벽히 일치했을 때의 비용 함수 값보다 더 크다.

