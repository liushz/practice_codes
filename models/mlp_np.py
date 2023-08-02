import numpy as np

# 定义MLP网络结构
input_size = 10
hidden_size = 20
output_size = 5

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化模型参数
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    # 定义损失函数
    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    # 定义激活函数
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 定义前向传播
    def forward(self, X):
        h = self.sigmoid(np.dot(X, self.W1) + self.b1)
        y_pred = np.dot(h, self.W2) + self.b2
        return y_pred, h

    # 定义反向传播和梯度下降更新参数
    def backward(self, X, y_true, y_pred, h):
        # 计算输出层梯度
        output_grad = (y_pred - y_true) / len(y_true)
        W2_grad = np.dot(h.T, output_grad)
        b2_grad = np.sum(output_grad, axis=0)
        # 计算隐藏层梯度
        hidden_grad = np.dot(output_grad, self.W2.T) * h * (1 - h)
        W1_grad = np.dot(X.T, hidden_grad)
        b1_grad = np.sum(hidden_grad, axis=0)

        # 更新参数
        self.W2 -= learning_rate * W2_grad
        self.b2 -= learning_rate * b2_grad
        self.W1 -= learning_rate * W1_grad
        self.b1 -= learning_rate * b1_grad


# 训练模型
X = np.random.randn(100, input_size)
y_true = np.random.randn(100, output_size)
learning_rate = 0.01
mlp = MLP(input_size, hidden_size, output_size)
for i in range(1000):
    y_pred, h = mlp.forward(X)
    loss = mlp.mse_loss(y_true, y_pred)
    mlp.backward(X, y_true, y_pred, h)
    if i % 100 == 0:
        print(f"Epoch {i}: Loss = {loss}")