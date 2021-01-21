### Date : 21.01.19
1. multy layer가 갖는 문제가 scale 변화나 변환이 되면 같은 건지 모른다고 했는데 cnn에서는 부분적으로 보기 때문에 그게 가능하다고 한다. 근데 data Augmentation를 해주는 이유가 무엇인가? 
- https://kevinthegrey.tistory.com/145
- 효과가 있구만
- overfitting 때문에 많이 씀
- 밝기조절, 이미지를 잘라주거나, 상하좌우 반전, 스케일을 커지게 해주거나 
[참고하기](https://nittaku.tistory.com/?page=45)

2. active func --> Relu 함수를 쓰면 back propagation할 때 문제가 없는건가????선형이라서 0이랑 max값만 나오는거 아닌가??
[activation back~](https://medium.com/@snaily16/what-why-and-which-activation-functions-b2bf748c0441)
[오차역전파](https://ratsgo.github.io/deep%20learning/2017/05/14/backprop/)
- activation function이 없으면 단순히 w*x+b의 합으로 이루어진 linear regression model이 된다.
- 왜 relu : sigmoid 랑 relu 두개다 편미분 해보면 왜 그런지 sigmoid 최대가 

3. batch optimizer (sgd)

Gradient Descent : Wupdate = Wprevious - learninglate * Gradient 방향
보폭을 얼마로 할지 learning rate

모델이 트레이닝 데이터 셋 전체에 대해 loss를 구하고 weight vector 업데이트
데이터셋 전체 loss :: batch gradient descent

##### BGD :: 전체 데이터 셋을 한번 다 본 후에 loss 를 구해서 weight 업데이트 
```python
for i in range(m):
	gradient=evaluate_gradient(training_data)
	weight=weight-learning_rate * gradient
```

##### SGD : 데이터 1개를 본 후에 바로 그거에 대해 weight update (learning rate 기준 한걸음 간 후) weight update >> 전체 데이터셋 확인 >> suffle >> 다시 반복  >>> 당연히 느리겟지!!!!!

```python
for i in range(m):
	np.random.shuffle(training_data)
	for one_data in training_data:
		gradient=evaluate_gradient(one_data)
		weight=weight-learning_rate * gradient
```

- tf.model.compile(optimization = 1!!! 요부분이다

#### Mini-batch 
- 데이터 B개를 본 후 평균 loss를 구하고 weight를 업데이트 해주면서 어떤 loss 함수의 진짜 최소값 찾기
```python
for i in range(m):
	np.random.shuffle(training_data)
	for one_batch in get_mini_batches(training_data,one_batch_size=32):
		gradient=evaluate_gradient(one_batch)
		weight=weight-learning_rate * gradient
```

- epoch : 전체 반복 횟수
- mini-batch size : 2의 제곱수로 적는 batch size
- Iteration : 전체 데이터의 수와 mini-batch size에 따라 자동 설정

4. 학습모델 저장과 vallidation_set 활용을 통해 overfitting 확인하기 
- MNIST 모델들은 잘 정규화 되있어서 MLP 모델에서도 괜찮은 성능을 보인 것!!!!
- MLP 공간적인 구조를 고려하지 않는다


5. [전이학습 (Transfer learning)](https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751)
- [한글](https://jeinalog.tistory.com/13)
- https://codingcrews.github.io/2019/01/22/transfer-learning-and-find-tuning/
- 
##### 이사람꺼 설명 좀 잘 되어있다. 고양이 개 합성곱 가중치 자연어 시계열 까지 보고 천천히 따라 해보자 
- https://codetorial.net/tensorflow/transfer_learning.html

#### 공식 페이지 전이학습도 나와있음
- https://www.tensorflow.org/tutorials/images/transfer_learning?hl=ko

##### 전이학습 pytorch 
- https://tutorials.pytorch.kr/beginner/transfer_learning_tutorial.html

##### 파이썬을 이용한 딥러닝 전이 학습 tensor 1.1,,,
- 예제 코드 https://github.com/wikibook/transfer-learning


##### objectdetection Yolo custom
- https://eehoeskrap.tistory.com/370




### D : 21.01.20 
오늘의 블로그 http://taewan.kim/post/cnn/
https://codetorial.net/tensorflow/transfer_learning.html

1. sigmoid 대신에 relu 쓰는 이유? :: 미분해보면 안다
- https://medium.com/@kmkgabia/ml-sigmoid-%EB%8C%80%EC%8B%A0-relu-%EC%83%81%ED%99%A9%EC%97%90-%EB%A7%9E%EB%8A%94-%ED%99%9C%EC%84%B1%ED%99%94-%ED%95%A8%EC%88%98-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0-c65f620ad6fd


2. learning late 조절하는 것 optimizer
- Adam 같은 것들 
- Adam은 learning rate를 처음에 미세하게 했다가 중간에는 많이 나중에는 다시 미세하게 조절함
- https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/


------------------------------------------

~/.pyenv/versions/tf2/lib/python3.7/site-packages/tensorflow/python/keras/utils$ sudo gedit data_utils.py 

```python
import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
```

추가


https://gaiag.tistory.com/75 : import 질문 함
http://taewan.kim/post/cnn/ : good CNN


-------------------------------------------------

### D : 21.01.21

0. 하고 싶은일
- 발표자료 완성 
- CNN 관련 딥러닝 글 더 읽으면서 정리하기
: 모델하나 따라 해보자!
- Catographer공부
- b에 정리
- ROS 한거 정리
- 일지적기(b)


1. 궁금한 거, 해결할 것
- MNIST 1에서 왜저렇게 튀는 건지,,,, > 가설 지역 구간에 들어갓는데 빠져나온거 아닌가? > 근데 전 과정을 기억하는 Adam이 아니라 minibatch를 썻으므로 불가능 할거 같은데
아니면 러닝레이트를 너무 작게 줘서 지역 구간에 빠진 거 일 수도 잇고 (Saddle point) 인가

- 각 레이어 풀링 그런거 계산법 : https://cs231n.github.io/convolutional-networks/
- pycharm 에서 그래프 어캐 그리냐,,
- ssl error 해결법 mnist


#### VGG네트워크 나 그런거 한번 따라해보려고 하다가
#####  작은 데이터셋으로 강력한 이미지 분류 모델 설계 [keraskorea](https://keraskorea.github.io/posts/2018-10-24-little_data_powerful_model/)


```
작은 네트워크를 처음부터 학습 (앞으로 사용할 방법의 평가 기준)
기존 네트워크의 병목특징 (bottleneck feature) 사용
기존 네트워크의 상단부 레이어 fine-tuning
```

+ 오늘 배울 케라스 기능
```
- ImageDataGenerator: 실시간 이미지 증가 (augmentation)
- flow: ImageDataGenerator 디버깅
- fit_generator: ImageDataGenerator를 이용한 케라스 모델 학습
- Sequential: 케라스 내 모델 설계 인터페이스
- keras.applications: 케라스에서 기존 이미지 분류 네트워크 불러오기
- 레이어 동결 (freezing)을 이용한 fine-tuning
```

```
$ pip install pillow
$ pip install scipy
```
pillow는 그림? 이미지 //  scipy는 수학!


1. [*Data 전처리와 Augmentation](https://keraskorea.github.io/posts/2018-10-24-little_data_powerful_model/)
- [Image generator class_Keras](https://keras.io/api/preprocessing/image/#imagedatagenerator-class)
- [Image 받는 곳](https://www.flickr.com/services/api/)
- [Image processing keras](https://keras.io/api/preprocessing/image/)
```python
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=40,      -> image회전 범위(degrees)
        width_shift_range=0.2,  -> 수평
        height_shift_range=0.2, -> 수직으로 랜덤하게 평행 이동 시키는 범위 (원본 가로 세로 길이의 비율 값)
        rescale=1./255,         -> 1./255 로 스케일링 하여 0 ~ 1로 정규화 --> 제일 먼저 적용
        shear_range=0.2,        -> shearing transoformation 범위
        zoom_range=0.2,         -> 임의 확대 / 축소
        horizontal_flip=True,   -> True 50% 확률로 이미지 뒤집
        fill_mode=`nearest`)    -> 이미지 회전 이동 축소 생길때 공간 채우는 거
```

- [pyenv 와 pycharm 활용](https://velog.io/@ssseungzz7/pyenv%EC%99%80-pyenv-virtualenv%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%9C-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EA%B0%9C%EB%B0%9C-%ED%99%98%EA%B2%BD-%EA%B5%AC%EC%84%B1%ED%95%98%EA%B8%B0-p3k15vjigb)


2. [CIFAR 10을 이용한 CNN](https://gruuuuu.github.io/machine-learning/cifar10-cnn/#)

- Normalization 전체 구간 [0, 1]
- Standardization
https://github.com/GRuuuuu/GRuuuuu.github.io/blob/master/assets/resources/machine-learning/CNN-CIFAR10/cifar10%20dropout%20test%20notebook-s.ipynb
https://gruuuuu.github.io/machine-learning/cifar10-cnn/#


3. VGG

- 내가 ssl error 해결한다 https://pip.pypa.io/en/latest/user_guide/#installing-from-wheels










#### 다 필요없고 pyenv는 굳이 할 필요가 없다 jupyter 쓰는게 아니라면 왜? pycharm을 쓰면 새로운 가상환경을 항상 만들 수 있고 그 위에 package를 install 할 수 있으니깐 그게 훨씬 편하다
NEW project 해서 New venv만들고 setting 가서 import해주는게 빠르고 쉬움



- 오 파이참에서는 run : ctrl+shift+F10해줘야 shift+f10쓸 수 있는듯
- 아 cuda 쓸려면 이런 설정이 필요하군 : https://www.tensorflow.org/install/source
- 근데 bazel 설치하다 보면 오류 날거임 :https://github.com/bazelbuild/bazel/issues/1512
```
sudo apt install curl gnupg
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list

---> echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list


```
s 빼주자
- openjdk 14 받음 : 18은 11 삭제 :https://codechacha.com/ko/ubuntu-install-open-jdk11/
- tensorflow anaconda https://velog.io/@somnode/linux-tensorflow-gpu-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0
- 내생각에 위에꺼 안해도 되는듯,,, 내가 그걸 못해서 저건 그냥 빌드 툴이고 bash 창에다 export 추가 안해줌 ㅋㅋ
- numa 경고 해결 방법 : https://hiseon.me/data-analytics/tensorflow/tensorflow-numa-node-error/
이해 잘 됨 -> https://aciddust.github.io/blog/post/Tip-NUMA-node-read-from-SysFS-had-negative-value-1/
- https://webnautes.tistory.com/1428  여기서 numa 무시해도 된데
- plt 해서 pycharm 에서 그림 뛰우려 했을 때 실패 했을 때 : https://www.programmersought.com/article/18335155927/

```
sudo apt-get install tcl-dev tk-dev python-tk python3-tk
```
- 사용
```python

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
```
- tk 이렇게 까는 거야,, : https://askubuntu.com/questions/815874/importerror-no-named-tkinter-please-install-the-python3-tk-package
- repo 추가하고 하는거,,
-  %ln -s /usr/lib/python3.8/lib-dynload/_tkinter.cpython-38-x86_64-linux-gnu.so _tkinter.cpython-38-x86_64-linux-gnu.so 


장소에서 인식하는데 뭔가 소리랑 연결해서 비가 오는 소리 사람 소리 뭐 그런걸로 낮밤 구별에 도움을 준다던가 그러면 안되는가?? Transformer??!!!


ppt dropout 그림 https://sacko.tistory.com/45




### 상반기 목표!!

지도 그리고 -- 해보고 큰 환경에서 slam 하고 navigation 공부
- slam에서 중요한 것 localization and sensor 임계치에 대한 이해 
  
- multy robot 서로 위치 공유 통신! 아마 군집-navigtaion : 세종시 ----

#### navigation plan 
<https://kcl-planning.github.io/ROSPlan/> ::ros pdd
<https://libraries.io/github/KCL-Planning/ROSPlan>
<https://kcl-planning.github.io/ROSPlan/documentation/?>

- 목표는 gazebo 큰환경에 다가 큰로봇으로 실외가능 로봇 카메라 (가자보에서 object detection- deep) -> 멀티 ->>>실제로 ->> Open manipulator??!! -- > husky or omo
- but problem : 논문거리,,,,
- 다음 주 : SLAM Navigation 공부
- ROS or ROS2 비교 해보고 ㄱㄱㄱㄱ

* 세부계획
```
1~2월 환경 셋업 
3 gazebo 환경
4~~실제 
```
--> 통신
--> 슬램을 합칠 수 있는 방법 > 지도 전송? 소켓 통신? 로스로 될 까?



- https://keraskorea.github.io/posts/2018-10-24-little_data_powerful_model/
- https://cs231n.github.io/convolutional-networks/
- https://gruuuuu.github.io/machine-learning/cifar10-cnn/#




mnist https://github.com/kairess/streamlit-examples



