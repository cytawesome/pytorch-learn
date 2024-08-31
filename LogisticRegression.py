import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionModel(torch.nn.Module): # Module Base class for all neural network modules
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, X):
        y_pred = F.sigmoid(self.linear(X)) #由于sigmoid（1/1+e^-x）不包含任何参数，不需要在init中初始化告诉模型有别的参数，所以只需要进行计算构建计算图
        return y_pred

def train():
    #train_set = torchvision.datasets.MNIST(root='../dataset/mnist', train=True, download=True)
    #test_set = torchvision.datasets.MNIST(root='../dataset/mnist', train=False, download=True)
    x_data = torch.tensor([[1.0], [2.0], [3.0]])
    y_data = torch.tensor([[0.0], [0.0], [1.0]])

    model = LogisticRegressionModel()
    criterion = torch.nn.BCELoss(size_average=True) #二元交叉熵
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epcho in range(5000):
        y_pred = model(x_data) #调用model.call,model.call是在父类里面，会调用forward方法
        loss = criterion(y_pred, y_data)
        print(epcho, loss.item())

        optimizer.zero_grad() #让所有权值的梯度为0
        loss.backward()
        optimizer.step() #梯度下降更新梯度

    print("w= ", model.linear.weight.item())
    print("b= ", model.linear.bias.item())

    x_test = torch.tensor([[4.]])
    y_test = model(x_test)
    print("y_pred = ", y_test.data)

    x = np.linspace(0, 10, 200)
    x_t = torch.tensor(x, dtype=torch.float32).view((200,1))
    with torch.no_grad():
        y_t = model(x_t)
    y = y_t.data.numpy()
    plt.plot(x, y)
    plt.plot([0, 10], [0.5, 0.5], c='r')
    plt.xlabel('Hours')
    plt.ylabel('Probability of Pass')
    plt.grid()
    plt.show()



if __name__ == '__main__':
    train()