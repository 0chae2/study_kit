*본 게시글은 개인적인 공부를 위해 작성된 내용임으로, 자세한 내용은 아래의 명시한 출처를 참고해주세요.



## 기존 Multi-layered Neural Network의 문제
 - Image에 변형(크기변화 혹은 이동) 시에 같은 이미지라고 판단하지 못 함
    + 글자의 topology는 고려하지 않고, raw data에 대해 직접적으로 처리를 하기 때문에 엄청나게 많은 학습 데이터 필요 , 학습시간 증가 문제
    + ex) 32 * 32 pont // black and white 패턴에 대해 처리를 하기 위해서는 2^(32*32) = 2^1024 >> 이것을  Glay-scale에 적용한다면 256^(32*32) = 256^1024개의 패턴나옴,,,,
 - 학습시간, 망의 크기, 변수의 개수의 문제!!!!
    
## [CNN이란](https://velog.io/@tmddn0311/CNN-tutorial)

- 도입배경 영상이 가지는 공간적 특성을 강화 시킴(local [receptivefield](https://distill.pub/2019/computing-receptive-fields/))
- 전체 영상에 대해 가중치 및 바이어스공유(shared parameter)
- 자유변수의 수 를 줄임(free parameter) --> CNN학습시간 줄임
- Overfitting 가능성 줄임
- 영상에서 잘 구별할 수 있는 특징(Salient feature)얻을 수 있음 
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
- Kernel : 한번에 처리할 노드의 크기

### layer
- 32 * 32 * 3 = width * height * channel
```
Most widely used for image classification
Generally, it consists of convolution layer, pooling and fully-connected layer
Convolution, Pooling layer - feature extraction
Fully-conneted layer - classification
```
1. Convolution layer
- feature maps / activation maps을 만드는 필터
- output feature maps의 채널 수 == filter 수
![con1](https://github.com/0chae2/study_kit/blob/main/Deep/CNN/pic/con1.png)

- 채널이 1개일때
![con1](https://github.com/0chae2/study_kit/blob/main/Deep/CNN/pic/con2.png)

- 채널이 3개일때
![con1](https://github.com/0chae2/study_kit/blob/main/Deep/CNN/pic/con3.png)
![con1](https://github.com/0chae2/study_kit/blob/main/Deep/CNN/pic/con4.png)

- 채널 계산이 끝난 후 activation func을 활용해서 계산을 해준다.
![con1](https://github.com/0chae2/study_kit/blob/main/Deep/CNN/pic/act.png)

2. [Pooling layer](https://supermemi.tistory.com/16)
- convolution 계산 결과인 feature maps에서 filter 특징을 더 뽑아내기 위해
- input size를 줄이거나
- overfitting을 조절 : parameter을 줄여 훈련데이터만 높은 성능을 보이는 과적합을 줄일 수 있다


- filter
![filter 출처https://www.youtube.com/watch?v=Em63mknbtWo&list=PLQ28Nx3M4Jrguyuwg4xe9d9t2XE639e5C&index=31](https://github.com/0chae2/study_kit/blob/main/Deep/CNN/pic/Filter.png)
- stride
![stride](https://github.com/0chae2/study_kit/blob/main/Deep/CNN/pic/stride.png)
![check](https://github.com/0chae2/study_kit/blob/main/Deep/CNN/pic/check.png)
- padding
1) 그림이 급격하게 작아지는 것을 방지
2) 이부분이 모서리다! 라는 것을 알려 줌

![padding](https://github.com/0chae2/study_kit/blob/main/Deep/CNN/pic/padding.png)

- Activation maps(?,?,filter 갯수) : ? ? > image 크기와 filter 크기 마다 달라짐
![activation](https://github.com/0chae2/study_kit/blob/main/Deep/CNN/pic/swiping.png)
![full](https://github.com/0chae2/study_kit/blob/main/Deep/CNN/pic/fully.png)

--------------------
### [CNN실습](https://www.youtube.com/watch?v=9fldE3-yJpg&list=PLQ28Nx3M4Jrguyuwg4xe9d9t2XE639e5C&index=34)

1. tf.keras.layer.Conv2D
```python
tf.keras.layers.Conv2D(
    filters, kernel_size, strides=(1, 1), padding='valid',
    data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
    use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, **kwargs
)
```
- filters : 사용할 필터 수
- kernel_size : int, tuple / list 다 가능 ex)  3 / (3,3) / [3,3]
- strides : integer / tuple / list
- data_format : channels_last - default (batch, height, width, channels) / channels_first (batch, channels, height, width)
- padding : valid P = 0 padding 을 안 해주는 / stride가 1일 때 기준으로 입력과 출력이 같아짐 Same (Case-insensitive)
![padding](https://github.com/0chae2/study_kit/blob/main/Deep/CNN/pic/p1.png)



2. [tf.keras.layers.MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D)
```python
tf.keras.layers.MaxPool2D( pool_size=(2, 2), strides=None, padding='valid', data_format=None, **kwargs )
```
- pool_size :  pooling에 사용할 filter의 크기를 정하는 것.(단순한 정수, 또는 튜플형태 (N,N))
- strides :  pooling에 사용할 filter의 strides를 정하는 것.
- padding :  "valide"(=padding을 안하는것) or "same"(=pooling결과 size가 input size와 동일하게 padding)

 
##### Input shape:
- If data_format='channels_last': 4D tensor with shape (batch_size, rows, cols, channels).
- If data_format='channels_first': 4D tensor with shape (batch_size, channels, rows, cols).
##### Output shape:
- If data_format='channels_last': 4D tensor with shape (batch_size, pooled_rows, pooled_cols, channels).
- If data_format='channels_first': 4D tensor with shape (batch_size, channels, pooled_rows, pooled_cols).
fit(
    x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
    validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
    sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_batch_size=None, validation_freq=1,
    max_queue_size=10, workers=1, use_multiprocessing=False
)





----------
### Modern CNN [Image classification](https://github.com/0chae2/study_kit/blob/main/Deep/CNN/imageclassification.md)
LeNet,
AlexNet,
VGG Nets,
GoogLeNet,fit(
    x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
    validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
    sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_batch_size=None, validation_freq=1,
    max_queue_size=10, workers=1, use_multiprocessing=False
)
ResNet
#### Image classification에서 자주 등장하는 [top-5 error and top-1 error](https://www.quora.com/What-does-the-terms-Top-1-and-Top-5-mean-in-the-context-of-Machine-Learning-research-papers-when-report-empirical-results)



### Image Detection (object detection)
RCNN,
Fast RCNN,
Faster RCNN,
SPP Net,
Yolo,
SDD,
Attention Net
### Semantic Segmentation
FCN, DeepLab v1, v2
U-Net,
ReSeg,
Image Captioning

##### [Image classification / object detection 차이](https://bskyvision.com/413)
1. Image classification
2. Object detection

> 다시 확인


#### [출처]
[1] [그림원출처](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk)<br>
[2] [함수설명]https://supermemi.tistory.com/16)
