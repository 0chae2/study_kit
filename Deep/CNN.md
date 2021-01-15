*개인적인 공부를 목적으로 출처표기 후 작성하였습니다.
1. Neural Network
- 인간의 뇌를 모방하여 만든 것
- Input * Weight -- activation function-->output
- Back propagation(역전파) : trainnig을 통해 weight를 결정해주는 것
  
  1) Activation Function
  - step function
  - sigmoid function
  - ReLU
  - Softmax function : multiclass classification 문제에서 많이 사용
    ![softmax]()(softmax)



2. CNN이란
- Fully connected와 차이점
    0) Input
    1) Feature extraction : 특징을 추출하기 위한 단계
    2) Shift and distortion invariance : topology 변화에 영향을 받지 않도록 해주는 단계
    3) Classification : 분류기
    4) Output
    
    
- Filter란 ? 
    + 특징이 데이터에 있는지 없는지 검출해 주는 함수
    + 각기다른 특징들을 검출해 줄 수 있는 것

- Stride : 필터를 적용하는 간격


-----------------------------------------------------------------
    
#### 용어정리

- Under fitting : 학습데이터가 모자라거나, 학습이 제대로 되지 않는 것, 트레이닝 데이터에 가깝게 가지 못 하는 경우

- Over fitting : 트레이닝 데이터에 그래프가 너무 정확히 맞아들어갈 때, 샘플 데이터에 너무 정확히 학습되어 있는 경우
    + over fitting sol ) 충분한 트레이닝 데이터를 준비한다 / 피처 수를 줄인다 / regularization정규화를 한다!
    





  
[참고한 자료]  
- [머신러닝개요](https://m.blog.naver.com/laonple/221166694845)
- [AlexNet](https://bskyvision.com/421)
- [조대협님 블로그](https://bcho.tistory.com/1149)
- [humkim git](https://github.com/hunkim/DeepLearningZeroToAll)
- [모두의 딥러닝2](https://www.youtube.com/watch?v=qPMeuL2LIqY&list=PLQ28Nx3M4Jrguyuwg4xe9d9t2XE639e5C&index=2)
- [라온피플딥러닝 개요](https://blog.naver.com/PostView.nhn?blogId=laonple&logNo=220608018546)
- [정리오짐](https://excelsior-cjh.tistory.com/79)
