*개인적인 공부를 목적으로 출처 표기 후 작성하였습니다.

🥝 찾아보자<br>

- [PCA](https://bskyvision.com/347?category=635506) : 주 성분분석
- [LDA](https://bskyvision.com/351?category=635506) : 선형판별분석

- MNIST 성능 그래프 따기
- CNN강의듣기
- CNN / Alexnet 뭐 그런거 정리
----------------------------------------------------------

#### 🥝 [Tensorflow2.0](https://github.com/0chae2/study_kit/blob/main/Deep/program.md)
#### 🍎 [CNN이론](https://github.com/0chae2/study_kit/blob/main/Deep/CNN/README.md)
#### [Deep learning confusion](https://programmersought.com/article/3136974038/) : matrix accuracy top1 top5 accuracy of each class

----------------------------------------------------------------

- Neural Network : 인간의 뇌를 모방하여 만든 것
- Input * Weight -- activation function-->output
 
#### 🍎 [MGD / SGD](https://light-tree.tistory.com/133)
- Batch : 일괄적으로 처리되는 집단 , Iteration 1회당 사용되는 training data set의 묶음
- Batch gradient descent(BGD) : 
- https://nonmeyet.tistory.com/entry/Batch-MiniBatch-Stochastic-%EC%A0%95%EC%9D%98%EC%99%80-%EC%84%A4%EB%AA%85-%EB%B0%8F-%EC%98%88%EC%8B%9C
- https://hcnoh.github.io/2018-11-27-batch-normalization
- https://sacko.tistory.com/category/Data%20Science/%EB%AC%B8%EA%B3%BC%EC%83%9D%EB%8F%84%20%EC%9D%B4%ED%95%B4%ED%95%98%EB%8A%94%20%EB%94%A5%EB%9F%AC%EB%8B%9D?page=2

#### 🍎 [batch와 epoch](https://bskyvision.com/803)
- batch : 집단한 무리, 한회분을 묶다 > 딥러닝에서는 모델의 가중치를 한번 업데이트 시킬 때 사용되는 샘플들의 묶음
    ex) 1000개 샘플 중 배치 사이즈가 20이라면 20개의 샘플 단위마다 모델의 가중치를 한번씩 업데이트 시킨다는 말, 즉 50번 가중치가 업데이트 된다는 말!!!! 하나의 데이터 셋을 50개의 배치로 나눠서 훈련을 진행했다고 보면 됨
- epoch : 중요한 사건, 변화들이 일어난 시대! > 딥러닝에서 에포크는 학습의 횟수를 의미한다 ex) epoch 10, batch 20 가중치를 50번 업데이트 하는 것을 총 10번 반복한다는!! 각 데이터 샘플이 총 10번씩 사용되는 것이다! 결과적으로 50 * 10 = 500번 업데이트
<br>
-https://nittaku.tistory.com/293


🍎 Back propagation(역전파) : trainnig을 통해 weight를 결정해주는 것  

🍎 Under fitting : 학습데이터가 모자라거나, 학습이 제대로 되지 않는 것, 트레이닝 데이터에 가깝게 가지 못 하는 경우

🍎 [Over fitting](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit) : 트레이닝 데이터에 그래프가 너무 정확히 맞아들어갈 때, 샘플 데이터에 너무 정확히 학습되어 있는 경우
    + over fitting sol ) 충분한 트레이닝 데이터를 준비한다 / 피처 수를 줄인다 / regularization정규화를 한다!<br>
    

-----------

### Activation Function

- step function
- sigmoid function
- tanh
- ReLU
- Softmax function : multiclass classification 문제에서 많이 사용

---------------------
### Hyperparameter

- learning rate : 오차를 학습에 얼마나 반영할 지![learning rate]()

- [cost function = loss function](http://www.gisdeveloper.co.kr/?p=7631)
  + Mean square Error (평균제곱오차)
  + Cross-Entropy Error(교차 엔트로피 오차)
  
- Regularization parameter(정규화)
- Mini-batch 크기 : N개의 이미지를 넣어서 학습 하는 법 ????
- Training 반복 횟수 : Training 횟수 너무 많으면 overfitting 
- Hidden unit 개수 : 많으면 네트워크 표현력 넓어져서 좋은 성능 낼 수도 있지만, overfitting 될 수도 있음 적으면 underfitting
- Weight intialization(가중치 초기화) : 모든 초기 값을 0으로 설정했을 떄 모든 뉴런이 동일한 결과를 내어, Back propagation 과정에서 동일한 gradient 값을 얻는다. 그렇게 되면 모든 파라미터가 동일한 값으로 update 되어 뉴런의 개수가 의미가 없어짐 *가중치는 보통 입력 데이터 수를 n으로 둘 때 +1/sqrt(n) ~ -1/sqrt(n)안에서 랜덤으로 결정함 


### Hyperparameter optimization

- Grid Search
- Random search
- Bayesian optimization
- SGD : Stochastic Gradient Descent 확률적 경사하강법
- Adam


-----------------------------------------------------------
##### [Numpy함수](https://codetorial.net/numpy/functions/index.html)



  
[참고한 자료]  
- [머신러닝개요](https://m.blog.naver.com/laonple/221166694845)
- [AlexNet](https://bskyvision.com/421)
- [조대협님 블로그](https://bcho.tistory.com/1149)
- [humkim git](https://github.com/hunkim/DeepLearningZeroToAll)
- [모두의 딥러닝2](https://www.youtube.com/watch?v=qPMeuL2LIqY&list=PLQ28Nx3M4Jrguyuwg4xe9d9t2XE639e5C&index=2)
- [모두의 딥러닝 깃허브](https://github.com/hunkim/DeepLearningZeroToAll/tree/master/tf2)
- [라온피플딥러닝 개요](https://blog.naver.com/PostView.nhn?blogId=laonple&logNo=220608018546)
- [AI note](https://github.com/SeonminKim1/AI_Notes)
[1][tensorflow good](https://codetorial.net/tensorflow/basics_of_optimizer.html)<br>
[2][Deep 개념](https://excelsior-cjh.tistory.com/79)<br>
[3][김태영의케라스 :케라스기본개념](https://tykimos.github.io/lecture/)<br>
[4][나중에 따라해](https://www.edwith.org/deeplearningai4/lecture/34895)<br>
[5][one-shot learing설명](https://medium.com/mathpresso/%EC%83%B4-%EB%84%A4%ED%8A%B8%EC%9B%8C%ED%81%AC%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EA%B2%80%EC%83%89%EA%B8%B0%EB%8A%A5-%EB%A7%8C%EB%93%A4%EA%B8%B0-f2af4f9e312a)<br>
[6][One-shot 발표](http://dsba.korea.ac.kr/seminar/?mod=document&uid=63)<br>
[7][밑바닥 부터 시작하는 딥러닝](https://velog.io/@jakeseo_me/%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0-%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94-%EB%94%A5%EB%9F%AC%EB%8B%9D-2-2-MNIST-%EC%86%90%EA%B8%80%EC%94%A8-%EC%88%AB%EC%9E%90-%EC%9D%B8%EC%8B%9D)<br>
[8][성능 평가까지 완벽](https://velog.io/@tmddn0311/mnist-classification)<br>
[][Keras API/선형회귀,로지스틱,다중입력 예제](https://wikidocs.net/38861)<br>
[][Keras Docs](https://keras.io/ko/optimizers/)<br>
[][간단한 동영상](https://www.youtube.com/watch?v=VWFPlPYxzNg&list=PLVNY1HnUlO2702hhjCldVCwKiudLHhPG0)<br>
[][앙상블부터 보기](https://ebbnflow.tistory.com/133)<br>
[][앙상블 정리](https://teddylee777.github.io/scikit-learn/scikit-learn-ensemble)<br>
[][tensorflow DQN](https://github.com/devsisters/DQN-tensorflow/)<br>
[][오차역전파이론](https://excelsior-cjh.tistory.com/171)<br>
[][Mnist여러가지모델로 최적화](https://buomsoo-kim.github.io/keras/2018/04/22/Easy-deep-learning-with-Keras-4.md/)<br>

[][Mnist 데이터의 특징](https://supermemi.tistory.com/10?category=835152)
[][batch 잘 정리](
  [해당깃]https://github.com/shuuki4/Batch-Normalization)
##### CNN
 - https://velog.io/@tmddn0311/CNN-tutorial
 - [CNN 오차역전파](https://ratsgo.github.io/deep%20learning/2017/04/05/CNNbackprop/)


[][markdown 사용법](https://steemit.com/kr/@buket47/emoji-2018-05-10)<br>
[][markdown 이모티콘 출처](http://www.iemoji.com/#?category=food-drink&version=36&theme=appl&skintone=default)<br>
[][sklearn](https://scikit-learn.org/stable/)<br>
[][Deep learning Image classification Guidebook](https://hoya012.github.io/blog/deeplearning-classification-guidebook-1/)



##### [tensorflow model](https://github.com/tensorflow/models/tree/master/research)
##### https://github.com/google-research/receptive_field
