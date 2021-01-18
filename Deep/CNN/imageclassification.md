*본 게시글은 개인적인 공부를 위해 작성된 내용임으로, 자세한 내용은 아래의 명시한 출처를 참고해주세요.
- LeNet, AlexNet, VGGNet, GoogLeNet, ResNet, SENet 
### LeNet-5 1998


### AlexNet, 2012
- GTX 580 2개로 연산 병렬처리
- Input 227 * 227 * 3 images
#### First layer(conv1) 
- 96개 11 * 11 filters applied at stride 4
- output : 55 * 55 * 96
- parameters : (11 * 11 * 3) * 96 = 35K

#### Second layer(Pool1)
- 3 * 3 filters applied at stride 2
- output : 27 * 27 * 96
- parameters : 0!
> norm 최근에 안함 
![alex1](https://github.com/0chae2/study_kit/blob/main/Deep/CNN/pic/alex1.png)


1. ReLu Nonlinearity
- tanh(2/(1+e^(-2x))-1)함수를 사용하는 것이 아닌 ReLu 함수를 활용함 
- ReLU : max(0,x) 를 처음 제안한 논문은 아님 
- speed : ReLU == tanh*6

2. Local Response Normalization(LRN)
- 신경생물학의 lateral inhibition

3. Overlapping Pooling
- 기존에는 Non-overlapping pooling 이였음
- kernel size를 stride보다 크게 overlapping pooling하는 방법 채택
  + Non : stride 2 >> kernel 2*2 maxpooling
  + overlapping : stride 1 >> kernel 2*2 maxpooling 
 
4. Dropout
- Overfitting 방지를 막기 위한 규제기술의 일종인 dropout을 사용했다. 
- 훈련 시(테스트 모든 뉴런 사용!) fully-connected layer 의 뉴런 중 일부를 생략하면서 학습을 진행하는 것, 몇개의 뉴런을 0으로 바꿈 >> 그 뉴런들은 forward pass 와 back propagation에 영향을 미치지 못함

- PCA
- data augmentation : 데이터의 변형을 통해 > 반전 or 크기변화 이동 > 하나의 이미지를 여러가지 학습셋으로 늘이는 역할 


### GoogleNet



-------------------------
#### 질문
- data augmentation:: 근데 이게 왜 되는 것인가? multi-layered neural network의 이 문제점을 해결한 것이 cnn아닌가 근데 이게 어떻게 학습셋을 늘이는 역할을 하지?
- overlapping pooling :: 이게 왜 더 학습률이 좋을까? > 학습을 겹치면서 시켜서???



[참고]
- https://hoya012.github.io/blog/deeplearning-classification-guidebook-1/
- https://bskyvision.com/418
- https://blog.naver.com/laonple/220648539191






[출처]
[1] https://hoya012.github.io/blog/deeplearning-classification-guidebook-1/

