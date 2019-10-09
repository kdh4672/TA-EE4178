
# PyTorch 개론 (+Google Colab 사용법)
<hr>

## 0. Basic Python

### 0.1 자료형과 연산

#### 숫자형


```python
a = 10
b = 1.2
print(a, b)
print(a+b, a-b, a*b, a/b)
```

    10 1.2
    11.2 8.8 12.0 8.333333333333334


#### 문자열


```python
quote = "Life is too short,"
print(quote, "Python is fast") 
```

    Life is too short, Python is fast


#### 리스트 / 튜플

* 리스트


```python
a = [1, 2, 'Life', ['a', 'b', 'c']]
```


```python
print(a[0]) 
```

    1



```python
print(a[-1]) 

```

    ['a', 'b', 'c']



```python
print(a[-1][1]) 
```

    b



```python
print(a[0:2]) 
```

    [1, 2]



```python
a.append('short') 
a.insert(2, 3)
print(a)
```

    [1, 2, 3, 'Life', ['a', 'b', 'c'], 'short']


* 튜플: 리스트와 비슷하지만 요소값 생성, 삭제, 수정이 불가

#### 딕셔너리 / 집합


```python
dic = {'key': 'value', '류현진': '야구', '손흥민': '축구'}
```


```python
print(dic['류현진']) 
```

    야구



```python
print(dic.keys())
```

    dict_keys(['key', '류현진', '손흥민'])



```python
print(dic.values())
```

    dict_values(['value', '야구', '축구'])



```python
dic['new key'] = 'new value' 
print(dic)
```

    {'key': 'value', '류현진': '야구', '손흥민': '축구', 'new key': 'new value'}


### 0.2 제어문

#### if 문


```python
a = [1,2,3]
if a[1]%2==0:
    print("{} is even number".format(a[1]))
elif a[1]%2==1:
    print("{} is odd number".format(a[1]))
else:
    print("Impossible")
```

    2 is even number


#### for 문


```python
for i in range(0, 3):
    print(i)
```

    0
    1
    2


#### while 문


```python
a = 0
while True:
    if a == 3:
        break
    print(a) 
    a +=1
```

    0
    1
    2


### 0.3 함수 / 클래스

#### 함수


```python
def square_sum(a, b):
    squared_a = a**2
    squared_b = b**2
    return squared_a + squared_b

c = square_sum(3, 4)
print(c)
```

    25


#### 클래스 


```python
class Calculator():
    def __init__(self):
        self.description = "Example for initializing a Class"
        self.result = 0

    def add(self, num1, numb2):
        if type(num1) == int and type(numb2) == int:
            self.result = self.result + num1 + numb2
            return self.result
        else:
            raise Exception

cal = Calculator()
print(cal.description)
print(cal.add(10, 20))
```

    Example for initializing a Class
    30


### 0.4 Packages

#### import


```python
import numpy as np
a = np.array([1,2,3])
print(a, type(a))
```

    [1 2 3] <class 'numpy.ndarray'>


<hr>

## 1. PyTorch

### 1.1 Tensors / Numpy

#### Tensors / Numpy

* Tensors


```python
import torch
```


```python
# 초기화 되지 않은 행렬 생성
x = torch.empty(5, 3)
print(x)
```

    tensor([[8.4078e-44, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00]])



```python
# 0~1 사이에서 랜덤으로 초기화된 행렬 생성
x = torch.rand(5, 3)
print(x)
```

    tensor([[0.9196, 0.1910, 0.9334],
            [0.6659, 0.5679, 0.7001],
            [0.9826, 0.2074, 0.0880],
            [0.9500, 0.8362, 0.7135],
            [0.2773, 0.4744, 0.9578]])



```python
# Standard Normal Distribution 에서 랜덤으로 초기화된 행렬 생성
x = torch.randn(5, 3)
print(x)
```

    tensor([[ 0.1538,  1.1824, -0.6295],
            [ 0.5266,  1.7692, -1.1491],
            [ 0.3576,  1.6991,  0.9910],
            [-1.2054, -0.1789, -0.6816],
            [ 0.0559,  0.5304, -0.8403]])



```python
# 값 지정하여 행렬 생성
x = torch.tensor([1.2, 2.4])
print(x)
```

    tensor([1.2000, 2.4000])


#### 1.2 PyTorch Project 구조
