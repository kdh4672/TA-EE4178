## PyTorch 개론 (+Google Colab 사용법)

### 0. Basic Python

##### 0.1 자료형과 연산

1. 숫자형

   ```python3
   a = 10
   b = 1.2
   print(a, b) #=> 10 1.2 11.2
   print(a+b, a-b, a*b, a/b) 
   #=> 11.2 8.8 12.0 8.33333333334
   ```

2. 문자열

   ```python3
   quote = "Life is too short,"
   print(quote +  "Python is fast") 
   #=> Life is too short, Python is fast
   ```

3. 리스트 / 튜플

   * List

   ```python3
   a = [1, 2, 'Life', ['a', 'b', 'c']]
   print(a[0]) 
   #=> 1
   print(a[-1]) 
   #=> ['a', 'b', 'c']
   print(a[-1][1]) 
   #=> ['b']
   print(a[0:2]) 
   #=> [1, 2]
   
   a.append('short') 
   #print(a) => [1, 2, 'Life', ['a', 'b', 'c'], 'short']
   a.insert(2, 3) 
   #print(a) => [1, 2, 3, 'Life', ['a', 'b', 'c'], 'short']
   ```

   * Tuple: 리스트와 비슷하지만 요소값 생성, 삭제, 수정이 불가

4. 딕셔너리 / 집합

   ```python3
   dic = {'key': 'value', '류현진': '야구', '손흥민': '축구'}
   print(dic['류현진']) 
   #=> '야구'
   print(dic.keys()) 
   #=> dict_keys(['key', '류현진', '손흥민'])
   print(dic.values()) 
   #=> dict_values(['value', '야구', '축구'])
   dic['new key'] = 'new value' 
   #print(dic) => {'key': 'value', '류현진': '야구', '손흥민': '축구', 'new key': 'new value'}
   ```

   

##### 0.2 제어문

1. if

   ```
   a = [1,2,3]
   if a[1]%2==0:
   	print("{} is even number".format(a[1]))
   elif a[1]%2==1:
   	print("{} is odd number".format(a[1]))
   else:
   	print("Impossible")
   #=> 2 is even number
   ```

2. for

   ```python3
   for i in range(0, 3):
   	print(i)
   #=> 0
   #=> 1
   #=> 2
   ```

3. while

   ```
   a = 0
   while True:
     if a == 3:
     	break
   	print(a) 
     a +=1
   #=> 0
   #=> 1
   #=> 2
   ```

   

##### 0.3 함수 / 클래스

1. 함수

   ```python3
   def square_sum(a, b):
   	squared_a = a**2
   	squared_b = b**2
   	return a+b
   	
   c = square_sum(3, 4)
   print(c)
   #=> 25
   ```

2. 클래스

   ```python3
   class Calculator():
   	def __init__(self):
   		self.description = "This class is example for initializing a Class"
   		self.result = 0
   	
   	def add(self, num1, numb2):
   		if type(num1) == int and type(numb2) == int:
   			self.result = self.result + num1 + numb2
   			return self.result
   		else:
   			raise Exception
   		
   cal = Calculator()
   print(cal.description)
   #=> This class is example for initializing a Class
   print(cal.add(10, 20))
   #=> 30
   ```



##### 0.5 Packages

* import

  ```python3
  import numpy as np
  a = np.array([1,2,3])
  print(a, type(a))
  #=> [1,2,3] <class 'numpy.ndarray'>
  ```

<br>



### 1. PyTorch

##### 1.1 Tensors / Numpy

1. Tensors / Numpy
   * Tensors
   
     ```python3
     import torch
     
     # 초기화 되지 않은 행렬 생성
     x = torch.empty(5, 3)
     
     # 0~1 사이에서 랜덤으로 초기화된 행렬 생성
     x = torch.rand(5, 3)
     
     # Standard Normal Distribution 에서 랜덤으로 초기화된 행렬 생성
     x = torch.randn(5, 3)
     
     # 값 지정하여 행렬 생성
     x = torch.tensor([1.2, 2.4])
     ```
   
   * Numpy
   
   * Tensors to Numpy / Numpy to Tensors
2. Tensors 연산

##### 1.2 PyTorch Project 구조

1. Load Data
2. Define Model
3. Set Loss & Optimizer
4. Train
5. Save & Visualization

<br>

### cf. Google Colab

[사용법](https://drive.google.com/file/d/11B7cjkW0KVMZv-yqxHDhg0TUE3CESYSx/view)

