### Date : 21.01.19
1. multy layer가 갖는 문제가 scale 변화나 변환이 되면 같은 건지 모른다고 했는데 cnn에서는 부분적으로 보기 때문에 그게 가능하다고 한다. 근데 data Agument를 해주는 이유가 무엇인가? 
- https://kevinthegrey.tistory.com/145
- 효과가 있구만


2. active func --> Relu 함수를 쓰면 back propagation할 때 문제가 없는건가????선형이라서 0이랑 max값만 나오는거 아닌가??
[activation back~](https://medium.com/@snaily16/what-why-and-which-activation-functions-b2bf748c0441)
[오차역전파](https://ratsgo.github.io/deep%20learning/2017/05/14/backprop/)
- activation function이 없으면 단순히 w*x+b의 합으로 이루어진 linear regression model이 된다.


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



