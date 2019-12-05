## EE 4178 (2019 Fall)

인공지능(딥러닝) 개론

* 교재: [Deep Learning with Python - Francois Chollet](https://www.pdfdrive.com/deep-learning-with-python-e54511249.html)
* 실습환경:
  * 언어: Python 3.6 (Python 3.5+) | [[점프 투 파이썬](https://wikidocs.net/book/1)] [[Python basic](https://wikidocs.net/book/1553)] [[파이썬 코딩도장](https://dojang.io/course/view.php?id=7)]
  * 딥러닝 프레임워크: PyTorch 1.3 (PyTorch 1.2+) | [[홈페이지](https://pytorch.org)]
  * 환경: Google Colab | [[홈페이지](https://colab.research.google.com/notebooks/welcome.ipynb)] [[사용법](https://drive.google.com/open?id=11B7cjkW0KVMZv-yqxHDhg0TUE3CESYSx)]

<br>

## Table of Contents

### 1. Basics

* [PyTorch 개론 (+Google Colab 사용법)](https://nbviewer.jupyter.org/github/gamchanr/TA-EE4178/blob/master/01-basics/intro_pytorch/intro_pytorch.ipynb) - [[Full Code](https://github.com/gamchanr/TA-EE4178/blob/master/01-basics/intro_pytorch/intro_pytorch.py)]
* [Binary Classification 모델 만들기 (XOR)](https://nbviewer.jupyter.org/github/gamchanr/TA-EE4178/blob/master/01-basics/classification/classification.ipynb) - [[Full Code](https://github.com/gamchanr/TA-EE4178/blob/master/01-basics/classification/binary_classification-xor.py)] | [IMDB] | [Face Recognizer]
* [Multi-class Classification 모델 만들기 (MNIST)](https://nbviewer.jupyter.org/github/gamchanr/TA-EE4178/blob/master/01-basics/classification/classification.ipynb#border1) - [[Full Code](https://github.com/gamchanr/TA-EE4178/blob/master/01-basics/classification/multiclass_classification-mnist.py)]
* Linear Regression 모델 만들기 (Boston Housing Price)
* Log Regression 모델 만들기

### 2. Intermediate

* [CNN(Convolutional Neural Network) (MNIST)](https://nbviewer.jupyter.org/github/gamchanr/TA-EE4178/blob/master/02-intermediate/CNN/cnn.ipynb?flush_cache=true) - [[Full Code - Train](https://github.com/gamchanr/TA-EE4178/blob/master/02-intermediate/CNN/cnn.py) / [Full Code - Test](https://github.com/gamchanr/TA-EE4178/blob/master/02-intermediate/CNN/test.py)] | [[CIFAR-10](https://github.com/gamchanr/TA-EE4178/blob/master/02-intermediate/CNN/cifar10.py)]
* RNN(Recurrent Neural Network)
* Stytle Transfer
* [VAE(Varialtional Auto-Encoder)](https://github.com/gamchanr/TA-EE4178/blob/master/02-intermediate/VAE/VAE.ipynb) - [[Full Code](https://github.com/gamchanr/TA-EE4178/blob/master/02-intermediate/VAE/train.py)]
* GAN(Generative Adversarial Networks)


### 3. Advanced
* Custom Dataloader  
* [Trasfer Learning (Using Pre-trained Model to Custom Case)]()

<!---
https://hackernoon.com/binary-face-classifier-using-pytorch-2d835ccb7816
https://m.blog.naver.com/PostView.nhn?blogId=gkvmsp&logNo=221485860027&proxyReferer=https%3A%2F%2Fwww.google.com%2F


- Custom Dataloader
- Paper Implementation
- Custom Modeling
- PyTorch for mobile

cf. Training Tips

- Train-Val-Test / Overfitting-Underfitting
- Data Augmentation

--->

<br>

## Final Project
1. [프로젝트개요](https://drive.google.com/open?id=1VYOuNUQQynr9hX2WcEqzAGGCBl5vukRH)
2. 데이터셋 - [[train](https://drive.google.com/open?id=1Gx-1Gj3YLR7r4kYIMDJMnF1GtKYPMvbQ)] / [[validation](https://drive.google.com/open?id=1T8KSOgAVpKsJFWgNMeVfLgTnKQSp1VeB)]
3. [데이터 로드를 위한 참고 코드 (font_dataset.py)](https://github.com/gamchanr/TA-EE4178/blob/master/shared/font_dataset.py)
