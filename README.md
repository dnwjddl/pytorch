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
- ```convolution Layer``` : 이미지 특징 추출
- ```pooling Layer``` : 필터를 거친 여러 특징 중 가장 중요한 특징 하나를 고르기 (덜 중요한 특징을 버리기 때문에 이미지 차원이 축소)
### [4] ResNet(CNN)
컨볼루션 커널을 여러겹 겹치며 복잡한 데이터에도 사용가능
1. [ResNet-CIFAR10](https://github.com/dnwjddl/pytorch-in-DeepLearning/blob/master/%5B4%5D%20ResNet.ipynb)

- shortcut 모듈은 증폭할때만 따로 갖게 됨

```python
class BasicBlock(nn.Module):
    ...
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(x)
        out = F.relu(out)
        return out
        
class ResNet(nn.Module):
    ...
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
     ...
```
### [5] Autoencoder
사람의 지도 없이 학습(encoder + decoder), 잠재변수 시각화, 잡음을 통하여 특징 추출 우선순위 확인  
1. [AutoEncoder](https://github.com/dnwjddl/pytorch-in-DeepLearning/blob/master/%5B5%5D%20AutoEncoder.ipynb)


### [6] RNN : 순차적 데이터 처리(영화 리뷰 감정 분석 & 기계 번역)
1. [GRU-TetClassification](https://github.com/dnwjddl/pytorch-in-DeepLearning/blob/master/%5B6%5D%20RNN_TextClassification.ipynb)
* tokenizing, word dictionary, word embedding
* RNN의 gradient vanishing을 해결하기 위하여 GRU 사용
    - ```update gate``` 이전 은닉 벡터가 지닌 정보를 새로운 은닉 벡터가 얼마나 유지할지
    - ```reset gate``` 새로운 입력이 이전 은닉 벡터와 어떻게 조합하는지 결정

2. [RNN-Seq2Seq](https://github.com/dnwjddl/pytorch-in-DeepLearning/blob/master/%5B6%5D%20RNN_Seq2Seq.ipynb)

- 인코더 RNN + 디코더 RNN

✔ **Teacher Forcing**

<img src="https://user-images.githubusercontent.com/72767245/105319929-e49e3780-5c08-11eb-82ab-175661217f4c.png" width="30%"> <img src="https://user-images.githubusercontent.com/72767245/105319969-f2ec5380-5c08-11eb-8a85-de6f5242069d.png" width="30%">

- 많은 데이터에서는 디코더가 예측한 토큰을 다음 반복에서 입력될 토큰으로 갱신해주는 것이 정석
- 하지만 학습이 아직 되지 않은 상태의 모델은 잘못된 예측 토큰을 입력으로 사용될 수 있으므로, **Teacher Forcing** 사용
- 디코더 학습 시 실제 번역문의 토큰을 디코더의 전 출력 값 대신 입력으로 사용해 학습을 가속하는 방법
- 번역문의 i번째 토큰에 해당하는 값 targets[i] 를 디코더의 입력값으로 설정

```python
def __init__(self, vocab_size, hidden_size):
    ...
    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.encoder = nn.GRU(hidden_size, hidden_size)
    self.decoder = nn.GRU(hidden_size, hidden_size)
    self.project = nn.Linear(hidden_size, vocab_size)
    
def forward(self, inputs, targets):
    ...
    embedding = self.embedding(inputs).unsqueeze(1) #임베딩
    encoder_output, encoder_state = self.encoder(embedding, initial_state)  
    # encoder_state: 문맥벡터 >> decoder에서 첫번째 은닉벡터로 쓰임
    # encoder_output: y >> 실제 output 값
    decoder_state = encoder_state
    ...
    decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
    
    # 디코더의 출력값으로 다음 글자 예측하기
    projection = self.project(decoder_output)
    outputs.append(projection)
    
    #피처 포싱을 이용한 디코더 입력 갱신
    decoder_input = torch.LongTensor([targets[i]])
    
    ...
```

### [7] 적대적 공격
FGSM 공격  
1.[AdversialAttack](https://github.com/dnwjddl/pytorch-in-DeepLearning/blob/master/%5B7%5D%20Adversial%20Attack.ipynb)


### [8] GAN
새로운 이미지 생성   
1. [cGAN](https://github.com/dnwjddl/pytorch-in-DeepLearning/blob/master/%5B8%5D%20GAN.ipynb)  

cGAN에 레이블 정보 추가
```python
# '진짜'와 '가짜' 레이블 생성
real_labels = torch.ones(BATCH_SIZE, 1)
fake_labels = torch.zeros(BATCH_SIZE, 1)
```
```python
#진짜와 가짜 이미지를 갖고 낸 오차를 더해서 판별자의 오차 계산한다
d_loss = criterion(D(images), real_labels) + criterion(D(G(z)), fake_labels)
```

### [9] DQN
게임환경에서 스스로 성장   
1. [DQN-cartpole](https://github.com/dnwjddl/pytorch-in-DeepLearning/blob/master/%5B9%5D%20DQN.ipynb)  
memory를 deque에 넣어서 오래된 경험(멍청할때의 경험)은 삭제됨  
act() 함수는 epsilon값을 넣어주어 무작위 행동을 하도록 함   
```self.model(state)```- 행동들에 대한 가치값  
```action```은 epsilon 합쳐져서 1,0 둘 중 하나 action 고름  
loss는 현재 가치값(정답)과 예측된 가치값(예측, 할인해줌)의 차이로 구해줌

```python
env= gym.make('CartPole-v0')

agent = DQNAgent()
for e in range(1, EPISODES+1):
    state= env.reset()
    ...
    while True:
        env.render() # 게임 화면 띄우기
```

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

✔ GRU 내 torch 연산 [iterator 연산]
- parameters() 함수는 그 신경망 모듈의 가중치 정보들을 iterator 형태로 반환하게 된다
- next(self.parameters() (그 외, iter(~), new()등)
```python
# nn.GRU 모듈의 첫 번째 가중치 텐서를 추출 > 이 텐서는 모델의 가중치텐서와 같은 데이터 타입
# new를 통하여 모델의 가중치와 같은 모양인 (n_layers, batch_size, hidden_dum) 모양의 텐서 변환
def _init_state(self, batch_size = 1):
    weight = next(self.parameter()).data
    return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
```


## 텐서 연산
### 간단 연산
- squeeze() & unsqueeze() : 1인 차원을 생성 or 제거
```squeeze()```함수를 사용하면 해당 인덱스의 값 squeeze  
```unsqueeze()```함수를 사용하면 해당 인덱스의 값에 1차원 추가
- contiguous()
연상과정에서 Tensor가 메모리에 올려진 순서 유지하려면 congiguous()사용
- transpose(), permute()
```transpose()```원본 tensor와 data를 공유하면서 새로운 tensor 반환
```permute()```는 모든 차원에서 맞교환할 수 있음(transpose()는 두 개의 차원을 맞교환)
- reshape(), view()
```reshape()```은 원본 tensor의 복사본 혹은 view를 반환한다

```transpose()```와 ```view()```의 차이점은 view함수는 오직 contiguous tensor에서만 작동, 반환하는 tensor 역시 contiguous하다  
transpose()는 non-contiguous와 contiguous tensor 둘다에서 작동 가능  
**contiguous 하고 싶다면 transpose().contiguous()해주어야함(permute도 동일)**

### 딥러닝 Dataset 
```python
### 데이터 로드 ###
trainset = datasets.FashionMNIST( root = './.data/', train = True,download = True, transform = transform)
```
```python
## transform: 텐서로 변환, 크기 조절(resize), 크롭(crop), 밝기(brightness), 대비(contrast)등 사용 가능
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])
```
```python
### DATALOADER ###
## DataLoader는 학습 이미지와 레이블을 (이미지, 레이블) 튜플 형태로 반환해줌
# Dataloader > 텐서화 & 정규화
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
## weight_decay 값이 커질수록 가중치 값이 작아지게 되고, Overfitting 현상은 해소 but underfitting 위험
# optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 0.0005)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.1)

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

✔ F.nll_loss(output, torch.tensor([263]))
- Negative log likelihood_loss
- ```F.logsoftmax``` + ```F.null_loss``` = ```F.crossEntropy```

✔ torch.clamp(input, min, max)
- 해당 범주 값 이상 이하가 되지 않도록 잡는 명령어

✔ **weight decay**
- L2 regularization은 가장 일반적으로 사용되는 regularization기법  
- 오버피팅은 가중치 매개변수의 값이 커서 발생하는 경우가 많기 때문에 가중치가 클 수록 큰 패널티를 부과
- L1 regularization도 동시에 사용가능

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
#### 검증 오차가 가장 적은 최적의 모델
```python
best_val_loss = None
for e in range(1, EPOCHS +1):
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, val_iter)
    
    # 검증 오차가 가장 적은 최적의 모델 저장 
    if not best_val_Loss or val_loss < best_val_loss:
        # 경로가 없을때 경로를 만들어줌
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        # 저장
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss
```
✔ K-Fold (교차 검증)
- 돌아가면서 모델을 훈련시켜 최적의 하이퍼파라미터를 찾는다

## 출처
펭귄브로의 3분 딥러닝 (파이토치 맛)
