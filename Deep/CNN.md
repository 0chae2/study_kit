*개인적인 공부를 목적으로 출처 표기 후 작성하였습니다.

----------------------------------------------------------------
    
#### 용어정리

- Under fitting : 학습데이터가 모자라거나, 학습이 제대로 되지 않는 것, 트레이닝 데이터에 가깝게 가지 못 하는 경우

- Over fitting : 트레이닝 데이터에 그래프가 너무 정확히 맞아들어갈 때, 샘플 데이터에 너무 정확히 학습되어 있는 경우
    + over fitting sol ) 충분한 트레이닝 데이터를 준비한다 / 피처 수를 줄인다 / regularization정규화를 한다!

-------
### Tensorflow 
- Tensor : 다차원 배열 (Multi-dimensional Array)
- 특징을 추출해주는 [Convoulution layer](https://tykimos.github.io/2017/01/27/CNN_Layer_Talk/)
1. [Keras Sequential](http://blog.daum.net/sualchi/13720852)
    + Keras의 Sequential 모델은 레이어들의 선형 스택(a linear stack of layers)로 되어있음
    ```python
    # model에 생성
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    model = Sequential([
    Dense(32, input_shape=(784,)), # 클래스32
    Activation('relu'),             # 선형 함수
    Dense(10),                       # 클래스 10
    Activation('softmax'),])
    
    # add() 활용 계층 추가
    model = Sequential()
    model.add(Dense(32, input_dim=784))
    model.add(Activation('relu'))
    model.add(Dense(10, input_dim=32)) #
    model.add(Activation('softmax'))    #
    ```
2. [tf.keras.models.Sequential.compile](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
```python
compile(
    optimizer='rmsprop', loss=None, metrics=None, loss_weights=None,
    weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs
)
```
    - optimizer :최적화모델 Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, Optimizer, RMSprop, SGD
    - loss : 
    - metrics :
    - loss_weights : 
    - weighted_metrics : 
    - run_eagerly :
    - steps_per_execution
    - **kwargs : 


###### [model생성](https://ebbnflow.tistory.com/128?category=738689)
- Sequential API : 단순한 층 쌓기 가능, 직관적 
- Functional API : 복잡한 층 쌓기 가능
##### [Tensor 기본](https://codetorial.net/tensorflow/basics_of_tensor.html)
```python
a = tf.constant(1) # constant는 상수 텐서를 만듬
b = tf.constant([2,3])
print(a) # tf.Tensor(1, shape=(), dtype=int32)
print(b) # tf.Tensor([2,3], shape=(2,),dtype=int32)

c = tf.zeros([2, 3]) #zero는 0으로 채워진 tensor 만듬 / ones는 1로 채워진 tensor 만듬
print(c)
tf.Tensor([[0. 0. 0.][0. 0. 0.]], shape=(2, 3), dtype=float32)
print(a.dtype, a.shape) # <dtype: 'int32'> 자료형 반환
```
##### [Numpy함수](https://codetorial.net/numpy/functions/index.html)

-----------------------------------


1. Neural Network
- 인간의 뇌를 모방하여 만든 것
- Input * Weight -- activation function-->output
- Back propagation(역전파) : trainnig을 통해 weight를 결정해주는 것
  
  1) Activation Function
  ```
    - step function
    - sigmoid function
    - ReLU
    - Softmax function : multiclass classification 문제에서 많이 사용
       ![softmax]()(softmax)
  ```
  2) Hyperparameter
  ```
    - learning rate : 오차를 학습에 얼마나 반영할 지![learning rate]()
    - cost function
      + Mean square Error (평균제곱오차)
      + Cross-Entropy Error(교차 엔트로피 오차)
    - Regularization parameter(정규화)
    - Mini-batch 크기
    - Training 반복 횟수 : Training 횟수 너무 많으면 overfitting 
    - Hidden unit 개수 : 많으면 네트워크 표현력 넓어져서 좋은 성능 낼 수도 있지만, overfitting 될 수도 있음 적으면 underfitting
    - Weight intialization(가중치 초기화) : 모든 초기 값을 0으로 설정했을 떄 모든 뉴런이 동일한 결과를 내어, Back propagation 과정에서 동일한 gradient 값을 얻는다. 그렇게 되면 모든 파라미터가 동일한 값으로 update 되어 뉴런의 개수가 의미가 없어짐 *가중치는 보통 입력 데이터 수를 n으로 둘 때 +1/sqrt(n) ~ -1/sqrt(n)안에서 랜덤으로 결정함 
  ```
  3) Hyperparameter optimization
  ```
    - Grid Search
    - Random search
    - Bayesian optimization
  ```

2. CNN이란
- Fully connected와 차이점

- Input
```
    1) Feature extraction : 특징을 추출하기 위한 단계
    2) Shift and distortion invariance : topology 변화에 영향을 받지 않도록 해주는 단계
    3) Classification : 분류기
```
- Output
    
    
- Filter란 ? 
    + 특징이 데이터에 있는지 없는지 검출해 주는 함수
    + 각기다른 특징들을 검출해 줄 수 있는 것

- Stride : 필터를 적용하는 간격


optimizer
- SGD : Stochastic Gradient Descent 확률적 경사하강법
- adam



  
[참고한 자료]  
- [머신러닝개요](https://m.blog.naver.com/laonple/221166694845)
- [AlexNet](https://bskyvision.com/421)
- [조대협님 블로그](https://bcho.tistory.com/1149)
- [humkim git](https://github.com/hunkim/DeepLearningZeroToAll)
- [모두의 딥러닝2](https://www.youtube.com/watch?v=qPMeuL2LIqY&list=PLQ28Nx3M4Jrguyuwg4xe9d9t2XE639e5C&index=2)
- [라온피플딥러닝 개요](https://blog.naver.com/PostView.nhn?blogId=laonple&logNo=220608018546)

[1][tensorflow good](https://codetorial.net/tensorflow/basics_of_optimizer.html)
[2][Deep 개념](https://excelsior-cjh.tistory.com/79)
[3][김태영의케라스](https://tykimos.github.io/lecture/)
[4][나중에 따라해](https://www.edwith.org/deeplearningai4/lecture/34895)
[5][one-shot learing설명](https://medium.com/mathpresso/%EC%83%B4-%EB%84%A4%ED%8A%B8%EC%9B%8C%ED%81%AC%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EA%B2%80%EC%83%89%EA%B8%B0%EB%8A%A5-%EB%A7%8C%EB%93%A4%EA%B8%B0-f2af4f9e312a)
[6][One-shot 발표](http://dsba.korea.ac.kr/seminar/?mod=document&uid=63)
[7][밑바닥 부터 시작하는 딥러닝](https://velog.io/@jakeseo_me/%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0-%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94-%EB%94%A5%EB%9F%AC%EB%8B%9D-2-2-MNIST-%EC%86%90%EA%B8%80%EC%94%A8-%EC%88%AB%EC%9E%90-%EC%9D%B8%EC%8B%9D)
[8][성능 평가까지 완벽](https://velog.io/@tmddn0311/mnist-classification)
[][Keras API/선형회귀그런거 정리](https://wikidocs.net/38861)
[][Keras Docs](https://keras.io/ko/optimizers/)
[][간단한 동영상](https://www.youtube.com/watch?v=VWFPlPYxzNg&list=PLVNY1HnUlO2702hhjCldVCwKiudLHhPG0)
[][앙상블부터 보기](https://ebbnflow.tistory.com/133)
[][앙상블 정리](https://teddylee777.github.io/scikit-learn/scikit-learn-ensemble)
[][tensorflow DQN](https://github.com/devsisters/DQN-tensorflow/)
[][오차역전파이론](https://excelsior-cjh.tistory.com/171)
[이거다!][Mnist여러가지모델로 최적화](https://buomsoo-kim.github.io/keras/2018/04/22/Easy-deep-learning-with-Keras-4.md/)
