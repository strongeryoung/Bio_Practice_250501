# 시그모이드 함수 시각화

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y, 'g')
plt.plot([0.0, 0.0], [1.0, 0.0], ':')  # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()

# 1. w값의 변화
def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5*x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)

plt.plot(x, y1, 'r', linestyle='--')  # w의 값이 0.5일 때
plt.plot(x, y2, 'g')   # w의 값이 1일 때
plt.plot([0.0, 0.0], [1.0, 0.0], ':')  # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()

# 2. b값의 변화
def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x+0.5)
y2 = sigmoid(x+1)
y3 = sigmoid(x+1.5)

plt.plot(x, y1, 'r', linestyle='--')   # x + 0.5
plt.plot(x, y2, 'g')  # x + 1
plt.plot(x, y3, 'b', linestyle='--')   # x + 1.5
plt.plot([0.0, 0.0], [1.0, 0.0], ':')  # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()



# 로지스틱 회귀 실습
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers


x = np.array([-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 숫자 10부터 1

model = Sequential()
model.add(Dense(1, input_dim=1, activation='sigmoid'))

sgd = optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])

model.fit(x, y, epochs=200)

plt.plot(x, model.predict(x), 'b', x, y, 'k.')


# # 예제: 10 이상은 1, 10 미만인 경우에는 0을 부여한 레이블 데이터
# x의 값이 5와 10사이의 어떤 값일 때 y값이 0.5가 넘기 시작하는 것으로 보임
# 정확도가 100%가 나왔었기 때문에 적어도 x의 값이 5일 때는 y값이 0.5보다 작고, x의 값이 10일 때는 y의 값이 0.5를 넘을 것
# x의 값이 5보다 작은 값일 때와 x의 값이 10보다 클 때에 대해서 y값을 확인

print(model.predict([1, 2, 3, 4, 4.5]))
print(model.predict([11, 21, 31, 41, 500]))