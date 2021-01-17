*본 게시글은 개인적인 공부를 위해 작성된 내용임으로, 자세한 내용은 아래의 명시한 출처를 참고해주세요.






## 기존 Multi-layered Neural Network의 문제
 - Image에 변형(크기변화 혹은 이동) 시에 같은 이미지라고 판단하지 못 함
    + 글자의 topology는 고려하지 않고, raw data에 대해 직접적으로 처리를 하기 때문에 엄청나게 많은 학습 데이터 필요 , 학습시간 증가 문제
    + ex) 32 * 32 pont // black and white 패턴에 대해 처리를 하기 위해서는 2^(32*32) = 2^1024 >> 이것을  Glay-scale에 적용한다면 256^(32*32) = 256^1024개의 패턴나옴,,,,
 - *학습시간, 망의 크기, 변수의 개수의 문제!!!!
    
## [CNN이란](https://velog.io/@tmddn0311/CNN-tutorial)
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


1. LeNet-5 1998


2. AlexNet, 2012
- GTX 580 2개로 연산 병렬처리


```

1. ReLu Nonlinearity
- tanh함 수를 사용하는 것이 아닌 ReLu 함수를 활용함 
- ReLU : max(0,x) 를 처음 제안한 논문은 아님 

2. Local Response Normalization

3. Overlapping Pooling
- 기존에는 Non-overlapping pooling 이였음
- kernel size를 stride보다 크게 overlapping pooling하는 방법 채택

4. 기타
- Dropout
- PCA
- data augmentation : 데이터의 변형을 통해 > 반전 or 크기변화 이동 > 하나의 이미지를 여러가지 학습셋으로 늘이는 역할 


```
#### 질문
- data augmentation:: 근데 이게 왜 되는 것인가? multi-layered neural network의 이 문제점을 해결한 것이 cnn아닌가 근데 이게 어떻게 학습셋을 늘이는 역할을 하지?
- overlapping pooling :: 이게 왜 더 학습률이 좋을까? > 학습을 겹치면서 시켜서???



[참고]
- https://hoya012.github.io/blog/deeplearning-classification-guidebook-1/
- https://bskyvision.com/418
- https://blog.naver.com/laonple/220648539191






[출처]
[1] https://hoya012.github.io/blog/deeplearning-classification-guidebook-1/

