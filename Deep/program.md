* 개인적인 공부를 위해 출처 표기 후 작성합니다!
### [Keras](https://keras.io/api/metrics/)
### Tensorflow 
- Tensor : 다차원 배열 (Multi-dimensional Array)
- 특징을 추출해주는 [Convoulution layer](https://tykimos.github.io/2017/01/27/CNN_Layer_Talk/)


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
-------------------------------
## Fully-Conneted
### 0. 데이터 셋 불러오기
### 1. 데이터 셋 전처리
- keras.utils.to_categorical : 원핫 인코딩! 7이라는 데이터가 있다면 --> 000000100로 변경해주는 것

### 2. [Keras Sequential](http://blog.daum.net/sualchi/13720852)
###### [model생성](https://ebbnflow.tistory.com/128?category=738689)
- Sequential API : 단순한 층 쌓기 가능, 직관적 
- Functional API : 복잡한 층 쌓기 가능

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
##### [🥑Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense?hl=ko)
- model.add(Dense(50, kernel_initializer='he_normal')) : he_normal :: It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in) where fan_in is the number of input units in the weight tensor.
- model.add(layers.Flatten()) : 
- model.add(Activation('sigmoid')) : activation 함수 이거 쓰겠다~

##### 🍇Activation
###### tf.keras.activations
1. sigmoid problem
- 극단 좌표계의 값들은 gradient 값이 매우 작게 됨 > deep한 네트워크 일 수록 >> 0에 가까워서 소실 될 수 있음 Vanising Gradient

2. Relu
- f(x) = max(0,x) >>> leaky relu 는 tf.keras.layers에 있음!


##### [🍇initializers](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal)
- Random : 평균 0, 분산 1
- Xavier Initialization : 평균 0, 분산 = 2/(channel_in + channel_out) 
- He Initialization : Relu함수 특화 평균 0, 분산 = 4/(channel_in + channel_out)
```python
tf.keras.initializers.RandomNormal()
tf.keras.initializers.glorot_uniform()
tf.keras.initializers.he_uniform()

```


### 3. [tf.keras.models.Sequential.compile](https://www.tensorflow.org/api_docs/python/tf/keras/Model)

 ```python
        compile( optimizer='rmsprop', loss=None, metrics=None, loss_weights=None,
         weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs)
 ```

#### [손실함수 평가지표](https://bskyvision.com/740?category=635506)
 - loss : 손실함수 훈련셋과 연관되어 훈련 시 사용
 - [metric](https://keras.io/api/metrics/accuracy_metrics/#accuracy-class) : 평가지표, 검증셋과 연관 훈련과정의 모니터링 시 사용됨
 
 
- optimizer :최적화모델 Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, Optimizer, RMSprop, SGD
- loss : 
- metrics :
- loss_weights : 
- weighted_metrics : 
- run_eagerly :
- steps_per_execution
- **kwargs : 

### 4. [tf.keras.models.Sequential.fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
- 오차역전파가 진행! loss function의 gradient가 역전파 > 그 gradient를 가지고 모델에게 맞는 최적의 가중치를 업데이트 하는 부분!
- tf.keras.models.fit
```python
fit(
    x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
    validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
    sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_batch_size=None, validation_freq=1,
    max_queue_size=10, workers=1, use_multiprocessing=False
)
```
- tf.keras.models.Sequential.fit
```python
fit(
    x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
    validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
    sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_batch_size=None, validation_freq=1,
    max_queue_size=10, workers=1, use_multiprocessing=False
)
```

### 모델 저장
https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
https://www.tensorflow.org/tutorials/quickstart/advanced?hl=ko

-----------------------------------

## CNN


