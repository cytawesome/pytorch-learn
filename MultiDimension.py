import torch
import numpy as np
import matplotlib.pyplot as plt
import time

class Model(torch.nn.Module): # Module Base class for all neural network modules
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 1) #8个feature，一个输出，y=x*w+b
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X):
        y_pred = self.sigmoid(self.linear(X)) #对于mini-batch（有n个x的时候），self.linear(x)得到的是一个矩阵(n*1)，sigmoid对矩阵的作用就是对矩阵中每一个元素sigmoid
        return y_pred

#simple neural network
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        #x = self.sigmoid(self.linear1(x))
        x = self.relu(self.linear1(x))
        #x = self.sigmoid(self.linear2(x))
        x = self.relu(self.linear2(x))
        x = self.sigmoid(self.linear3(x)) #最后这里输出y_pred不能是relu，因为relu在输入小于0时输出为0，这个在loss中log(0)会报错
        return x

def train(total_epcho):
    xy = np.loadtxt('./dataset/diabetes.csv.gz', delimiter=',', dtype=np.float32)

    device = torch.device('cpu')
    #device = torch.device('mps') #不知为啥，gpu速度远低于cpu速度，可能模型太简单了
    #device = torch.device('cuda')

    x_data = torch.from_numpy(xy[:, :-1]).to(device)
    y_data = torch.from_numpy(xy[:, -1:]).to(device) #非in-place操作

    model = NeuralNetwork()
    model.to(device) #in-place操作
    criterion = torch.nn.BCELoss(size_average=True) #二元交叉熵
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    draw_y = np.zeros(total_epcho)

    start_time = time.time() # 记录开始时间
    for epcho in range(total_epcho):
        y_pred = model(x_data) #调用model.call,model.call是在父类里面，会调用forward方法
        loss = criterion(y_pred, y_data)
        if epcho%(total_epcho//10) == 0 or epcho == total_epcho-1: #打印10次loss
            print(epcho, loss.item())
        draw_y[epcho] = loss.item() #记录loss

        optimizer.zero_grad() #让所有权值的梯度为0
        loss.backward()
        optimizer.step() #梯度下降更新梯度

    end_time = time.time() # 记录结束时间
    # 计算执行时间
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")


    #作图
    draw_x = np.linspace(0, total_epcho, total_epcho)

    plt.plot(draw_x, draw_y)
    plt.xlabel('epcho')
    plt.ylabel('loss')
    plt.grid()
    plt.show()

    #predict,check the number of false result and correct result
    with torch.no_grad():
        y_pred = model(x_data).cpu().numpy()

    y_data = y_data.cpu().numpy()
    total_correct = 0
    for i in range(len(y_pred)):
        if y_pred[i][0] < 0.5:
            y_pred[i][0] = 0
        else:
            y_pred[i][0] = 1
        if y_data[i][0] == y_pred[i][0]:
            total_correct += 1
    print("Prediction: total correct is {}, total wrong is {}, correction rate is {}".format(total_correct, len(y_pred) - total_correct, total_correct / len(y_pred)))

    #print("w= ", model.linear.weight.item())
    #print("b= ", model.linear.bias.item())

    #x_test = torch.tensor([[4.]])
    #y_test = model(x_test)
    #print("y_pred = ", y_test.data)

def test():
    # 检查是否有可用的 GPU
    if torch.backends.mps.is_available():
        print("MPS is available")
    else:
        print("MPS is not available")

if __name__ == '__main__':
    train(1000000)
    #test()