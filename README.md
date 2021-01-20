# ALGORITHM IN PYTORCH 
## 목차
#### [1] ANNetwork.ipynb)  
1. [ANN-tensor](https://github.com/dnwjddl/pytorch-in-DeepLearning/blob/master/%5B1%5D%20ANN_tensor.ipynb)
2. [ANN-Autograd](https://github.com/dnwjddl/pytorch-in-DeepLearning/blob/master/%5B1%5D%20ANN_Autograd.ipynb)  
3. [ANN-NeuralNetwork](https://github.com/dnwjddl/pytorch-in-DeepLearning/blob/master/%5B1%5D%20ANN_NeualNetwork.ipynb)  
#### [2] DNN - data augmentation과 dropout을 이용한 성능 높이기(과적합 줄이기)
1. [DNN-FashionMNIST](https://github.com/dnwjddl/pytorch-in-DeepLearning/blob/master/%5B2%5D%20DNN_FashionMNIST.ipynb)

#### [3] CNN - convolution filter을 사용한 이미지 처리
#### [4] ResNet(CNN) - 컨볼루션 커널을 여러겹 겹치며 복잡한 데이터에도 사용가능
- shortcut 모듈은 증폭할때만 따로 갖게 됨
#### [5] Autoencoder - 사람의 지도 없이 학습(encoder + decoder), 잠재변수 시각화, 잡음을 통하여 특징 추출 우선순위 확인
#### [6] RNN : 순차적 데이터 처리(영화 리뷰 감정 분석 & 기계 번역)
#### [7] 적대적 공격: FGSM 공격
#### [8] GAN : 새로운 이미지 생성
#### [9] DQN : 게임환경에서 스스로 성장
Seq2Seq, Adversarial Attack IN PYTORCH

## 텐서 생성
```python
torch.Tensor([[1,2,3],[4,5,6]], dtype = torch.int32, device = device)

# torch에서 dtype 나 device 바꾸는 방법
y = torch.as_tensor(x, dtype = torch.half, device = 'cpu')
y = x.to(device, dtype = torch.float.64)

# start부터 end까지 step별로 tensor
torch.arange(start = 0, end, step = 1, dtype = None, requires_grad = True)

torch.from_numpy() #numpy array인 ndarray로부터 텐서를 만듦

# N(0,1) 정규분포를 따르는 random 함수 추출
torch.randn(*sizes, dtype = , device = , requires_grad = )
```
```torch.Tensor```의 Autograd패키지는 모든 연산에 대해 **자동미분**을 제공  
텐서의 속성 중 하나인 ```x.requires_grad = True```로 하면 텐서의 모든 연산에 대해서 추적을 시작  
계산 작업 후 ```x.backward()```를 호출하면 모든 그레이디언트를 자동으로 계산할 수 있도록 함
<br><br>
텐서에 대한 기록(history) 추적 중지하려면 ```x.detach()``` 호출  
: 현재 계산 기록으로부터 분리시키고 이후 일어나는 계산들은 추적되지 않는다.  
<br><br>
기록 추적(및 메로리 사용)에 대해 방지를 하려면 ```with.no_grad()```를 wrap 가능
이 텐서의 변화도는 ```x.grad```속성에 누적된다  
: with.no_grad()는 변화도(gradient)는 필요없지만 ```requires_grad = True``` 가 설정되어 학습가능한 매개변수를 갖는 모델은 평가할 때 유용
<br><br>
```x.data```도 ```x.detach()```와 비슷한 기능
<br><br>
```x.detach()```: 기존 텐서에서 gradient가 전파가 안되는 텐서  
```x.clone()```: 기존 텐서와 내용 복사한 tensor 생성
<br><br>
```torch.new_tensor(x, requires_grad=True)```와 ```x.clone().detach().requires_grad(True)```는 같은 의미

## 텐서 연산

## 딥러닝 텐서
```python
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.02..)
loss = criterion(model(x_test).squeeze(), y_test)

optimizer.zero_grad() #optimizer 초기화

train_loss.backward() #그레이디언트 계산
optimizer.step() #step별로 계산
```

## 텐서 저장
학습된 모델을 state_dict() 함수 형태로 바꾸어준 후 .pt 파일로 저장  
state_dict() 함수는 모델 내 가중치들이 딕셔너리 형태로 {연산 이름: 가중치 텐서와 편향 텐서} 와 같이 표현된 데이터
```python
# 학습된 가중치 저장
torch.save(model.state_dict(), './model.pt')

# 이후 로드 [전이학습]
new_model = NeuralNet(2,5)
new_model.load_state_dict(torch.load('./model.pt'))
```

### ✔ sklearn의 make_blobs  
**분류용 가상 데이터 생성** : 등방성 가우시안 정규분포를 이용해 가상 데이터를 생성한다 (등방성: 모든 방향으로 같은 성질을 가진다는 뜻)
```python
from sklearn.datasets import make_blobs
x_train, y_train = make_blobs(n_samples = 80, n_features = 2,
                            centers = [[1,1],[1,-1],[-1,1],[-1,-1]],
                            shuffle = True, cluster_std = 0.3)
```      
[딥러닝 주의](https://www.notion.so/8-d72569a210ff489f9242ff74a831e5a4)

## 출처
펭귄브로의 3분 딥러닝 (파이토치 맛)
