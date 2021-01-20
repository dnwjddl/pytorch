# ALGORITHM IN PYTORCH 
## 목차
### [1] ANN
1. [ANN-tensor](https://github.com/dnwjddl/pytorch-in-DeepLearning/blob/master/%5B1%5D%20ANN_tensor.ipynb)
2. [ANN-Autograd](https://github.com/dnwjddl/pytorch-in-DeepLearning/blob/master/%5B1%5D%20ANN_Autograd.ipynb)  
3. [ANN-NeuralNetwork](https://github.com/dnwjddl/pytorch-in-DeepLearning/blob/master/%5B1%5D%20ANN_NeualNetwork.ipynb)  
### [2] DNN
data augmentation과 dropout을 이용한 성능 높이기(과적합 줄이기)
1. [DNN-FashionMNIST](https://github.com/dnwjddl/pytorch-in-DeepLearning/blob/master/%5B2%5D%20DNN_FashionMNIST.ipynb)
### [3] CNN
convolution filter을 사용한 이미지 처리
1. [CNN-FashionMNIST](https://github.com/dnwjddl/pytorch-in-DeepLearning/blob/master/%5B3%5D%20CNN.ipynb)
### [4] ResNet(CNN)
컨볼루션 커널을 여러겹 겹치며 복잡한 데이터에도 사용가능
- shortcut 모듈은 증폭할때만 따로 갖게 됨
1. [ResNet-CIFAR10](https://github.com/dnwjddl/pytorch-in-DeepLearning/blob/master/%5B4%5D%20ResNet.ipynb)
### [5] Autoencoder
사람의 지도 없이 학습(encoder + decoder), 잠재변수 시각화, 잡음을 통하여 특징 추출 우선순위 확인
[AutoEncoder](https://github.com/dnwjddl/pytorch-in-DeepLearning/blob/master/%5B5%5D%20AutoEncoder.ipynb)
### [6] RNN : 순차적 데이터 처리(영화 리뷰 감정 분석 & 기계 번역)
1. [RNN-TestClassification](https://github.com/dnwjddl/pytorch-in-DeepLearning/blob/master/%5B6%5D%20RNN_TextClassification.ipynb)
2. [RNN-Seq2Seq](https://github.com/dnwjddl/pytorch-in-DeepLearning/blob/master/%5B6%5D%20RNN_Seq2Seq.ipynb)
### [7] 적대적 공격: FGSM 공격
### [8] GAN : 새로운 이미지 생성
### [9] DQN : 게임환경에서 스스로 성장
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
### 간단 연산
- squeeze() & unsqueeze()
- congiguous()
- transpose(), permute()?

### 딥러닝 Dataset 
```python
### 데이터 로드 ###
trainset = datasets.FashionMNIST( root = './.data/', train = True,download = True, transform = transform)
```
```python
## transform: 텐서로 변환, 크기 조절(resize), 크롭(crop), 밝기(brightness), 대비(contrast)등 사용 가능
transform = transforms.Compose([
    transforms.ToTensor()
])
```
```python
### DATALOADER ###
train_loader = data.DataLoader(
    dataset= trainset,
    batch_size = batch_size )
```

✔ iter(), next() 
- for문을 사용하지 않고 iter() 함수를 사용하여 반복문 안에서 이용할 수 있도록 함
- next() 함수를 이용하여 배치 1개를 가지고 옴

```python
dataiter = iter(train_loader)
images, labels = next(dataiter)
```
✔ 시각화 (make_grid())
- ```img = utils.make_grid(images, padding=0)``` : 여러 이미지를 모아 하나의 이미지로 만들 수 있음
- 파이토치의 텐서에서 numpy로 바꾸어야 시각화 가능


## 딥러닝 텐서 과정
```python
### TRAIN ###
optimizer = torch.optim.SGD(model.parameters(), lr = 0.02..)

criterion = nn.BCELoss()
loss = criterion(model(x_test).squeeze(), y_test)
# loss = F.cross_entropy(output, target)

optimizer.zero_grad() #optimizer 초기화

train_loss.backward() #그레이디언트 계산
optimizer.step() #step별로 계산

### EVALUATE ###
with torch.no_grad():
  for data, target in test_loader:
    output = model(data)
    test_loss += F.cross_entropy(output, target, reduction='sum').item()
    
    pred = output.max(1, keepdim = True)[1]
    correct += pred.eq(target.view_as(pred)).sum().item()
```
✔ torch.max(1) -> 1차원을 기준으로 max 값을 정함  
ex) x = tensor(2,40,8)차원 -> x.max(1) -> 40 x 2(하나는 인덱스값, 하나는 값)

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

### 성능 올리기
#### 조기 종료
- 학습 중간중간 검증용 데이터셋으로 모델이 학습 데이터에만 과적합되지 않았는지 확인
- **검증 데이터셋에 대한 성능이 나빠지기 시작하기 직전이 가장 적합한 모델**
#### Data Augmentation
```python
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./.data',
                  train = True,
                  download = True,
                  transform = transforms.Compose([
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize((0.1307,),(0.3081,))
                  ])),
    batch_size = BATCH_SIZE, shuffle = True)
```
#### dropout
- 사실 layer가 얕은거 아니면 dropout보단 batch normalization이 더 나음
```python
## nn.Dropout & F.dropout 
# forward 함수 내
x = F.dropout(x, training = self.training, p = self.dropout

# main 함수 내
model = Net(dropout_p = 0.2)
```

## 출처
펭귄브로의 3분 딥러닝 (파이토치 맛)
