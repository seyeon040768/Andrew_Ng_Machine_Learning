# [Coursera Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning)
- [Week1](#week1)
  - [Introduction](#introduction)
    - [1. What is Machine Learning?](#1-what-is-machine-learning) 
    - [2. Supervised Learning](#2-supervised-learning)
    - [3. Unsupervised Learning](#3-unsupervised-learning)
  - [Model and Cost Function](#model-and-cost-function)
    - [1. Model Representation](#1-model-representation)
    - [2. Cost Function](#2-cost-function)
    - [3. Cost Function Intuition 1](#3-cost-function-intuition-1)
    - [4. Cost Function Intuition 2](#4-cost-function-intuition-2)
  - [Parameter Learning](#parameter-learning)
    - [1. Gradient Descent](#1-gradient-descent)
    - [2. Gradient Descent Intuition 1](#2-gradient-descent-intuition-1)
    - [3. Gradient Descent Intuition 2](#3-gradient-descent-intuition-2)
  - [Linear Algebra Review](#linear-algebra-review)
    - [1. Matrices and Vectors](#1-matrices-and-vectors)
    - [2. Addition and Scalar Multiplication](#2-addition-and-scalar-multiplication)
    - [3. Matrix Vector Multiplication](#3-matrix-vector-multiplication)
    - [4. Matrix Matrix Multiplication](#4-matrix-matrix-multiplication)
    - [5. Matrix Multiplication Properties](#5-matrix-multiplication-properties)
    - [6. Inverse and Transpose](#6-inverse-and-transpose)
- [Week2](#week2)
  - [Multivariate Linear Regression](#Multivariate-linear-regression)
    - [1. Multiple Features](#1-multiple-features)
    - [2. Gradient Descent for Multiple Variables](#2-gradient-descent-for-multiple-variables)
    - [3. Gradient Descent in Practice 1 - Feature Scaling](#3-gradient-descent-in-practice-1---feature-scaling)
    - [4. Gradient Descent in Practice 2 - Learning Rate](#4-gradient-descent-in-practice-2---learning-rate)
    - [5. Features and Polynomial Regression](#5-features-and-polynomial-regression)
- [Week3](#week3)
  - [Classification and Representation](#classification-and-Representation)
    - [1. Classification](#1-classification)
    - [2. Hypothesis Representation](#2-hypothesis-representation)
    - [3. Decision Boundary](#3-decision-boundary)
  - [Logistic Regression Model](#logistic-regression-model)
    - [1. Cost Function](#1-cost-function)
    - [2. Simplified Cost Function and Gradient Descent](#2-simplified-cost-function-and-gradient-descent)
    - [3. Advanced Optimization](#3-advanced-optimization)
  - [Multiclass Classification](#multiclass-classification)
    - [1. Multiclass Classification: One-vs-all](#1-multiclass-classification-one-vs-all)

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
<img src="https://latex.codecogs.com/gif.latex?\frac{1}{2\times3}\sum_{i=1}^{3}(h_{\Theta}(x_{i})-y_{i})^{2}=\frac{1}{6}(0.5^{2}+1^{2}+1.5^{2})=\frac{1}{6}\times\frac{7}{2}\simeq0.58" />   
과 같이 약 <img src="https://latex.codecogs.com/gif.latex?0.58" />이 나온다. 아까 데이터와 그래프가 완벽히 일치했을 때의 비용 함수 값보다 더 크다.

<img src="https://latex.codecogs.com/gif.latex?x" />값에 따른 비용 함수의 값을 좌표평면 위에 나타내면 아래와 같은 개형의 그래프가 그려진다.   

<img src="./Week1/cost_function_3.png" width="50%">

***
### 4. Cost Function Intuition 2

<img src="./Week1/cost_function_4.png" width="50%"><img src="./Week1/cost_function_5.png" width="50%"><img src="./Week1/cost_function_7.png" width="50%">

그래프와 데이터의 분포가 비슷할수록 등고선 그래프의 나타난 비용 함수의 값이 최하점에 가까워지는 것을 볼 수 있다.
***
## Parameter Learning
### 1. Gradient Descent
비용 함수의 값을 최소화하기 위해 사용하는 방법중에는 Gradient Descent(경사 하강법)이 있다.  
경사 하강법은 그래프의 최소값을 찾기 위해 말 그대로 경사를 따라 내려가는 방식이다.   
<img src="./Week1/gradient_descent_path_1.png" width="50%"><img src="./Week1/gradient_descent_path_2.png" width="50%">

위 그림과 같이 시작점에 따라 도착하는 지점이 다를 수 있다.

<img src="./Week1/gradient_descent.png" width="50%">

경사 하강법의 식은 위와 같으며 <img src="https://latex.codecogs.com/gif.latex?\Theta_{0}" />과 <img src="https://latex.codecogs.com/gif.latex?\Theta_{1}" />에 대해 따로 계산(편미분)하며 최소값에 수렴할 때 까지 반복한다.   
여기서 <img psrc="https://latex.codecogs.com/gif.latex?\alpha" />를 Learning Rate(학습률)이라 하고 학습률의 크기에 따라 한번에 내려가는 거리가 결정된다.

<img src="./Week1/simultaneous_update.png" width="50%">

경사 하강법을 계산할 때는 위와 같이 <img src="https://latex.codecogs.com/gif.latex?\Theta_{0}" />과 <img src="https://latex.codecogs.com/gif.latex?\Theta_{1}" />에 대한 값을 미리 계산한 다음에 대입하여야 한다. 오른쪽과 같이 계산 - 대입 - 계산 - 대입 순으로 계산하면 이상한 값이 나올 수도 있다.
***
### 2. Gradient Descent Intuition 1
<img src="./Week1/gradient_descent_start_right.png" width="30%"><img src="./Week1/gradient_descent_start_left.png" width="30%">

시작점이 최소값의 오른쪽일 때는 기울기가 양수이기 때문에 왼쪽으로 이동하게 되고 반대로 왼쪽일 때는 기울기가 음수이기 때문에 오른쪽으로 이동하게 된다.

<img src="./Week1/gradient_descent_LR_small.png" width="30%"><img src="./Week1/gradient_descent_LR_large.png" width="30%">

만약 학습률이 너무 작다면 조금씩 이동하기 때문에 최소값을 찾는데 너무 오래걸리게 된다.   
반대로 학습률이 너무 크다면 최소값으로 가지 못하고 오히려 멀어지게 된다.

<img src="./Week1/gradient_descent_LR_fixed.png" width="30%">

최소값에 가까워질수록 기울기가 0에 가까워지기 때문에 한번에 이동하는 거리가 짧아진다. 따라서 하강하는 도중 학습률을 수정(조정)할 필요가 없다.
***
### 3. Gradient Descent Intuition 2
앞에서 봤던 비용 함수   
<img src="https://latex.codecogs.com/gif.latex?J(\Theta_{0},\Theta_{1})=\sum_{i=1}^{m}(h_{\Theta}(x_{i})-y_{i})^{2}" />   
와 경사하강법   
<img src="./Week1/gradient_descent.png" width="50%">   
을 결합하면

<img src="https://latex.codecogs.com/gif.latex?\Theta_{0}:=\Theta_{0}-\alpha\frac{d}{d\Theta_{0}}(\frac{1}{2m}\sum_{i=1}^{m}(h_{\Theta}(x_{i})-y_{i})^{2})\\=\Theta_{0}-\alpha\frac{d}{d\Theta_{0}}(\frac{1}{2m}\sum_{i=1}^{m}(\Theta_{0}+\Theta_{1}x_{i}-y_{i})^{2})\\=\Theta_{0}-\alpha\frac{d}{d\Theta_{0}}(\frac{1}{2m}\sum_{i=1}^{m}(\Theta_{0}^{2}+2(\Theta_{1}x_{i}-y_{i})\Theta_{0}+(\Theta_{1}x_{i})^2-2\Theta_{1}x_{i}y_{i}+y_{i}^{2}))\\=\Theta_{0}-\alpha\frac{1}{2m}\sum_{i=1}^{m}(2\Theta_{0}+2(\Theta_{1}x_{i}-y_{i}))\\=\Theta_{0}-\alpha\frac{1}{m}\sum_{i=1}^{m}(\Theta_{0}+\Theta_{1}x_{i}-y_{i})\\=\Theta_{0}-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_{\Theta}(x_{i})-y_{i})" />

(<img src="https://latex.codecogs.com/gif.latex?\Theta_{0}" />에 대한 계산(미분))

이므로 아래와 같은 식이 나온다.

<img src="./Week1/gradient_descent_cost_function.png" width="50%">

<img src="https://latex.codecogs.com/gif.latex?\Theta_{1}" />에 대해 계산해도 위와 같은 식이 나온다.

***
## Linear Algebra Review
### 1. Matrices and Vectors
Matrix(행렬): 수를 직사각형 모양으로 배열한 것

<img src="./Week1/matrix_4_2.png" width="20%">

<img src="https://latex.codecogs.com/gif.latex?A_{ij}" />: <img src="https://latex.codecogs.com/gif.latex?i" />번째 열의 <img src="https://latex.codecogs.com/gif.latex?j" />번째 행에 위치한 원소   
ex) <img src="https://latex.codecogs.com/gif.latex?A_{32}" />: <img src="https://latex.codecogs.com/gif.latex?1437" />   
<img src="https://latex.codecogs.com/gif.latex?A_{24}" />: <img src="https://latex.codecogs.com/gif.latex?undefined" />   

Vector(백터): <img src="https://latex.codecogs.com/gif.latex?n\times1" />모양의 행렬

<img src="./Week1/vector_4.png" width="20%">

<img src="https://latex.codecogs.com/gif.latex?y_{i}" />: <img src="https://latex.codecogs.com/gif.latex?i" />번째 열에 위치한 원소   
ex) <img src="https://latex.codecogs.com/gif.latex?y_{2}" />: <img src="https://latex.codecogs.com/gif.latex?232" />
***
### 2. Addition and Scalar Multiplication
**행렬의 덧셈(행렬 + 행렬)**

<img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}a&b\\c&d\\e&f\end{bmatrix}+\begin{bmatrix}g&h\\i&j\\k&l\end{bmatrix}=\begin{bmatrix}a+g&b+h\\c+i&d+j\\e+k&f+l\end{bmatrix}" />

ex)

<img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}1&0\\2&5\\3&1\end{bmatrix}+\begin{bmatrix}4&0.5\\2&5\\0&1\end{bmatrix}=\begin{bmatrix}5&0.5\\4&10\\3&2\end{bmatrix}" />

**행렬의 곱셈(상수 X 행렬)**

<img src="https://latex.codecogs.com/gif.latex?x\times\begin{bmatrix}a&b\\c&d\\e&f\end{bmatrix}=\begin{bmatrix}ax&bx\\cx&dx\\ex&fx\end{bmatrix}" />

ex)

<img src="https://latex.codecogs.com/gif.latex?3\times\begin{bmatrix}1&0\\2&5\\3&1\end{bmatrix}=\begin{bmatrix}3&0\\6&15\\9&3\end{bmatrix}" />

***
### 3. Matrix Vector Multiplication
**행렬 X 백터**

<img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}a&b&c\end{bmatrix}\times\begin{bmatrix}x\\y\\z\end{bmatrix}=\begin{bmatrix}ax+by+cy\end{bmatrix}" />

ex)

<img src="./Week1/M_V_multi_ex.png" width="30%">

**집 가격 예측 예제**

<img src="./Week1/M_V_multi_ex_house.PNG" width="15%">

위와 같은 데이터가 있고

<img src="./Week1/M_V_multi_ex_house_h.PNG" width="30%">

위와 같이 <img src="https://latex.codecogs.com/gif.latex?\Theta_{0}" />과 <img src="https://latex.codecogs.com/gif.latex?\Theta_{1}" />을 <img src="https://latex.codecogs.com/gif.latex?-40" />과 <img src="https://latex.codecogs.com/gif.latex?0.25" />로 설정했을 때 아래와 같이 나타낼 수 있다.

<img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}1&2104\\1&1416\\1&1534\\1&852\end{bmatrix}\times\begin{bmatrix}-40\\0.25\end{bmatrix}" />

***
### 4. Matrix Matrix Multiplication
**행렬 X 행렬**

<img src="./Week1/M_M_mult.png" width="30%">

ex)

<img src="./Week1/M_M_multi_ex.png" width="30%">

<img src="https://latex.codecogs.com/gif.latex?m{\times}n" /> 모양의 행렬과 곱하려면 <img src="https://latex.codecogs.com/gif.latex?n{\times}o" /> 모양의 행렬이어야 한다. 이때 결과는 <img src="https://latex.codecogs.com/gif.latex?m{\times}o" /> 모양의 행렬이 나온다.

**집 가격 예측 예제**

<img src="./Week1/M_M_multi_ex_house.png" width="30%">

다음과 같은 데이터와 가설들이 있을 때, 아래와 같이 계산할 수 있다.

<img src="./Week1/M_M_multi_ex_house_result.png" width="30%">

***
### 5. Matrix Multiplication Properties
**교환법칙(commutative property)**

두 행렬 <img src="https://latex.codecogs.com/gif.latex?A" />와 <img src="https://latex.codecogs.com/gif.latex?B" />가 있을 때, <img src="https://latex.codecogs.com/gif.latex?A{\cdot}B{\neq}B{\cdot}A" />이다.(교환법칙이 성립하지 않는다.)

ex)

<img src="./Week1/M_multi_com_1.png" width="30%">

<img src="./Week1/M_multi_com_2.png" width="30%">

**결합법칙(associated law)**

행렬 <img src="https://latex.codecogs.com/gif.latex?A" />, <img src="https://latex.codecogs.com/gif.latex?B" />와 <img src="https://latex.codecogs.com/gif.latex?C" />가 있을 때, <img src="https://latex.codecogs.com/gif.latex?(A{\cdot}B){\cdot}C=A{\cdot}(B{\cdot}C)" />이다.(결합법칙이 성립한다.)

ex)

<img src="./Week1/M_multi_ass.png" width="30%">

**항등행렬(identity matrix)**

<img src="./Week1/M_multi_i.png" width="30%">

위와 같이 행과 열의 수가 같고 왼쪽 위부터 오른쪽 아래를 잇는 대각선(주대각선)에 있는 원소가 모두 1이고 나머지 원소는 0인 행렬을 항등행렬이라고 한다.
***
### 6. Inverse and Transpose
**역행렬(Inverse Matrix)**

행렬 <img src="https://latex.codecogs.com/gif.latex?A" />와 곱했을 때 항등행렬이 나오는 행렬을 <img src="https://latex.codecogs.com/gif.latex?A" />의 역행렬이라 하고 <img src="https://latex.codecogs.com/gif.latex?A^{-1}" />와 같이 표현한다.

ex)

<img src="./Week1/InT_inverse_ex.png" width="30%">

**전치행렬(Transposed Matrix)**

행렬 <img src="https://latex.codecogs.com/gif.latex?A" />의 행과 열을 맞바꾼 행렬을 <img src="https://latex.codecogs.com/gif.latex?A" />의 전치행렬이라 하고 <img src="https://latex.codecogs.com/gif.latex?A^{T}" />와 같이 표현한다.

ex)

<img src="./Week1/InT_A.png" width="15%"><img src="./Week1/InT_AT.png" width="15%">

행렬 <img src="https://latex.codecogs.com/gif.latex?A^{T}" />를 행렬 <img src="https://latex.codecogs.com/gif.latex?A" />의 전치행렬이라고 한다.   
주대각선을 기준으로 서로 대칭을 이룬다.
***

# Week2
## Multivariate Linear Regression
### 1. Multiple Features

<img src="./Week2/M_F_data.png" width="50%">

<img src="https://latex.codecogs.com/gif.latex?n" />: 특징(feature)의 수  ex) <img src="https://latex.codecogs.com/gif.latex?n=4" /> Price는 <img src="https://latex.codecogs.com/gif.latex?y" />   
<img src="https://latex.codecogs.com/gif.latex?x^{(i)}" />: <img src="https://latex.codecogs.com/gif.latex?i" />번째 데이터(학습 예제)  ex) <img src="https://latex.codecogs.com/gif.latex?x^{(2)}" />: <img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}1416&3&2&40\end{bmatrix}" />   
<img src="https://latex.codecogs.com/gif.latex?x^{(i)}_{j}" />: <img src="https://latex.codecogs.com/gif.latex?i" />번째 데이터(학습 예제)의 <img src="https://latex.codecogs.com/gif.latex?j" />번째 특징의 값  ex) <img src="https://latex.codecogs.com/gif.latex?x^{(3)}_{1}" />: <img src="https://latex.codecogs.com/gif.latex?1534" />

특징이 많을 때는 가설을 아래와 같이 표현한다.

<img src="./Week2/M_F_h_1.png" width="30%">

여기서 계산과 표기를 쉽게하기 위해 <img src="https://latex.codecogs.com/gif.latex?x_{0}=0" />을 설정한다.

<img src="./Week2/M_F_h_2.png" width="30%">

***
### 2. Gradient Descent for Multiple Variables

이전에 경사하강법(Gradient Descent)을 아래와 같이 표현했었는데,

<img src="./Week1/gradient_descent.png" width="30%">

변수(특징)가 여러개일 때는 아래와 같이 표현한다.

<img src="./Week2/Gradient_Descent_Multi_Var.png" width="30%">

<img src="./Week2/Gradient_Descent_Multi_Var_simple.png" width="30%">

***
### 3. Gradient Descent in Practice 1 - Feature Scaling
경사하강법의 계산 속도를 증가시키기 위해 Feature Scaling을 진행한다.   
Feature Scaling은 <img src="https://latex.codecogs.com/gif.latex?x" />값을 <img src="https://latex.codecogs.com/gif.latex?-1{\leq}x{\leq}1" />이나 <img src="https://latex.codecogs.com/gif.latex?-0.5{\leq}x{\leq}0.5" /> 사이로 만든다.

계산식은 아래와 같다.

<img src="https://latex.codecogs.com/gif.latex?x_{i}:=\frac{x_{i}-\mu_{i}}{s_{i}}" />

<img src="https://latex.codecogs.com/gif.latex?\mu_{i}" />: <img src="https://latex.codecogs.com/gif.latex?x" />값들의 평균   
<img src="https://latex.codecogs.com/gif.latex?s_{i}" />: <img src="https://latex.codecogs.com/gif.latex?max-min" /> 또는 표준편차
***
### 4. Gradient Descent in Practice 2 - Learning Rate
만약 <img src="https://latex.codecogs.com/gif.latex?\alpha" />(Learning Rate)이 너무 크다면 반복할 때마다 비용함수의 값이 증가한다.   
만약 <img src="https://latex.codecogs.com/gif.latex?\alpha" />(Learning Rate)이 너무 작다면 반복할 때마다 비용함수의 값이 매우 조금씩 감소한다.

만약 <img src="https://latex.codecogs.com/gif.latex?\alpha" />가 충분히 작다면 반복할 때마다 비용함수의 값이 계속 감소한다.
***
### 5. Features and Polynomial Regression

<img src="./Week2/polynomial_regression_graph.png" width="30%">

위와 같은 데이터가 있을 때, 직선(일차함수) 모양의 가설로는 정확한 예측을 하기 어렵다.   
이때 아래와 같은 가설을 사용할 수 있다.

<img src="./Week2/polynomial_regression_h.png" width="30%">

여기서 <img src="https://latex.codecogs.com/gif.latex?x^{2}" />이나 <img src="https://latex.codecogs.com/gif.latex?x^{3}" />은 Size를 제곱, 세제곱한 값이다.
***

# Week3
## Classification and Representation
### 1. Classification
분류는 어떤 데이터를 여러 값중 하나로 분류하는 것이다.

ex)
스팸 메일 분류, 온라인 거래 사기 유무, 종양 악성 유무

**Classification with Linear Regression**

<img src="./Week3/classification_tumor_1.png" width="30%">

위와 같은 데이터에 선형 회귀(Linear Regression)을 적용하면 그림과 같은 그래프가 나온다.   
<img src="https://latex.codecogs.com/gif.latex?y=0.5" />인 지점을 기준으로 앞은 양성종양, 뒤는 악성종양으로 분류하면 꽤 괜찮은 것 같이 보인다. 하지만 아래와 같은 그림을 보면 결과가 달라진다.

<img src="./Week3/classification_tumor_2.png" width="30%">

아까와 같이 <img src="https://latex.codecogs.com/gif.latex?y=0.5" />인 지점을 기준으로 나누면 문제가 생긴다. 

이와 같이 분류(Classification) 문제는 선형 회귀를 통해 해결하기에는 무리가 있다.(일부 데이터에서는 정상적으로 작동할 수 있어도 대부분은 잘 작동하지 않을 것이다.)
***
### 2. Hypothesis Representation

Logistic Regression에서는 아래와 같은 가설을 사용한다.

<img src="./Week3/hypothesis_representation_sigmoid.png" width="20%">

여기서 맨 아래에 있는 식은 Sigmoid Function 또는 Logistic Function이라고 하며 개형은 아래와 같다.

<img src="./Week3/hypothesis_representation_sigmoid_graph.png" width="30%">

<img src="./Week3/hypothesis_representation_p.png" width="30%">

가설을 위와 같이 표현할 수도 있는데, <img src="https://latex.codecogs.com/gif.latex?P(y=0|x;\theta)" />는 쉽게 말해 <img src="https://latex.codecogs.com/gif.latex?y" />가 <img src="https://latex.codecogs.com/gif.latex?0" />인 확률을 의미한다.

***
# 3. Decision Boundary

<img src="./Week3/hypothesis_representation_sigmoid_graph.png" width="30%">

만약 <img src="https://latex.codecogs.com/gif.latex?h_{\Theta}(x)\geq0.5" />이면 <img src="https://latex.codecogs.com/gif.latex?y=1" />이고,   
만약 <img src="https://latex.codecogs.com/gif.latex?h_{\Theta}(x)<0.5" />이면 <img src="https://latex.codecogs.com/gif.latex?y=0" />이다.

**Decision Boundary(Linear)**

<img src="./Week3/decision_boundary_graph_1.png" width="30%"><img src="./Week3/decision_boundary_h_1.png" width="30%">

위와 같은 데이터와 가설이 있을 때, <img src="https://latex.codecogs.com/gif.latex?\theta_{0}=-3,\theta_{1}=1,\theta_{2}=1" />이라고 하면 <img src="https://latex.codecogs.com/gif.latex?-3+x_{1}+x_{2}\geq0" />일 때, <img src="https://latex.codecogs.com/gif.latex?y=1" />이 된다.

예를 들어 <img src="https://latex.codecogs.com/gif.latex?x_{1}=1,x_{2}=2" />이면 <img src="https://latex.codecogs.com/gif.latex?-1<0" />이므로 <img src="https://latex.codecogs.com/gif.latex?y=0" />이 된다. 실제로 그림에서 확인해보면 <img src="https://latex.codecogs.com/gif.latex?(1,1)" /> 지점은 그래프아래에 위치하는것을 볼 수 있다.

**Decision Boundary(Non-linear)**

<img src="./Week3/decision_boundary_graph_2.png" width="30%"><img src="./Week3/decision_boundary_h_2.png" width="30%">

이번에는 직선으로 두 데이터를 나누기는 어려워 보인다. 이럴때에는 Polynomial Regression에서 했던것 처럼 위와 같이 가설을 만들어 주면 된다. 

<img src="https://latex.codecogs.com/gif.latex?\theta_{0}=-1,\theta_{1}=0,\theta_{2}=0,\theta_{3}=1,\theta_{4}=1" />와 같이 설정해 주면  <img src="https://latex.codecogs.com/gif.latex?-1+x_{1}^{2}+x_{2}^{2}\geq0" />일 때, <img src="https://latex.codecogs.com/gif.latex?y=1" />이 된다.   
여기서 앞에 나온 식은 반지름이 1이고 중심이 <img src="https://latex.codecogs.com/gif.latex?(0,0)" />인 원의 방정식이다.

이렇게 데이터를 나누는 경계선을 **Decision Boundary**라고 하며, 이는 가설에 의해 결정된다는 것을 알 수 있다.
***

## Logistic Regression Model
### 1. Cost Function

<img src="./Week3/cost_function.png" width="30%">

Logistic Regression의 비용함수는 위와 같다.   
<img src="https://latex.codecogs.com/gif.latex?y" />에 따라 식이 달라지는데 그 이유는 아래 그래프를 보면 된다.

<img src="./Week3/cost_function_graph.png" width="30%">

<img src="https://latex.codecogs.com/gif.latex?(0,0)" />을 지나는 그래프가 <img src="https://latex.codecogs.com/gif.latex?y=0" />의 적용되는 비용함수이고 다른 그래프는 그 반대이다.

가설 <img src="https://latex.codecogs.com/gif.latex?h(x)" />의 값은 시그모이드 함수를 통해 정해지기 때문에 무조건 <img src="https://latex.codecogs.com/gif.latex?0" />과 <img src="https://latex.codecogs.com/gif.latex?1" /> 사이의 값을 가진다. 

만약 결과(<img src="https://latex.codecogs.com/gif.latex?y" />)가 <img src="https://latex.codecogs.com/gif.latex?0" />일때, 가설의 값이 <img src="https://latex.codecogs.com/gif.latex?0" />에 가까울수록(정확할수록) 작아지고(<img src="https://latex.codecogs.com/gif.latex?0" />에 가까워지며 <img src="https://latex.codecogs.com/gif.latex?x=0" />이외에는 양수이다.) 가설의 값이 <img src="https://latex.codecogs.com/gif.latex?1" />에 가까울수록(부정확할수록) 기하급수적으로 커진다. 

***

### 2. Simplified Cost Function and Gradient Descent

<img src="./Week3/cost_function_simplified.png" width="30%">   
<img src="./Week3/cost_function_simplified_full.png" width="30%">

비용함수를 간단하게 표현하면 위와 같다. <img src="https://latex.codecogs.com/gif.latex?y" />값이 <img src="https://latex.codecogs.com/gif.latex?0" />이나 <img src="https://latex.codecogs.com/gif.latex?1" />일때에 때라 두 항중 하나의 항이 사라지게 된다.

vectorized한 식은 아래와 같다.

<img src="./Week3/cost_function_vectorized.png" width="30%">

비용함수를 미분한 식은 아래와 같은데,

<img src="./Week3/gradient_descent.png" width="30%">

놀랍게도 선형회귀의 비용함수를 미분한 식과 같다.

***

### 3. Advanced Optimization

경사하강법 외에도 conjugate gradient, BFGS, L-BFGS와 같은 여러 알고리즘들이 존재한다.

이러한 알고리즘들은 경사하강법보다 빠르고 학습률을 정해주지 않아도 되지만 복잡하다는 단점이 있다.
***

## Multiclass Classification
### 1. Multiclass Classification: One-vs-all

결과값이 <img src="https://latex.codecogs.com/gif.latex?0" />과 <img src="https://latex.codecogs.com/gif.latex?1" /> 중에서만 나오는 이진 분류와 달리 그 3개 이상의 결과값이 나오는 경우가 있을 수도 있다.

ex)   
이메일 분류 : 업무, 친구, 가족, 취미

<img src="./Week3/multiclass_classification_graph.png" width="30%">

위와 같이 결과값이 3개인 경우에는 각 class와 나머지 class를 나누어서 가설을 3개 만들면 된다.   
나중에 예측할 때에는 3개의 가설을 모두 테스트 해본 뒤에 가장 큰 값이 나온 가설에 해당하는 class일 확률이 높다고 판단하면 된다.(시그모이드 함수를 통해 계산하기 때문에 가장 큰 수일 수록 1에 가깝다. 즉 확률이 높다는 뜻이다.)

이러한 방식을 One-vs-all 방식이라고 한다.

수식으로 표현하면 아래와 같다.

<img src="https://latex.codecogs.com/gif.latex?\max_{i}(h_{\theta}^{(i)}(x))" />
***
