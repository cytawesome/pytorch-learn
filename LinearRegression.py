import torch

class Test():
    def __init__(self):
        pass

    def __call__(self, *args):
        print("Ha Ha "+str(args[0]))

def testCall():
    test = Test()
    test(1,2,3,4)

def testFunc():
    def func(*args, **kwargs):
        print(args)
        print(kwargs)
    func(1,2,3,4, x=1, y=2)

class LinearModel(torch.nn.Module): # Module Base class for all neural network modules
    def __init__(self):
        super(LinearModel, self).__init__() #这是python2调用父类构造方法的代码，python3用super().__init__()
        self.linear = torch.nn.Linear(1, 1) #这个模型是[[y1],[y2],...[yn]]=[[x1],[x2],...[xn]]*w+b的形式，纵向是n个数据，水平方向是feature数量

    #def __call__(self, *args): #想当于这样
    #    return self.forward(args[0])

    def forward(self, X): #前馈要进行的计算,overwrite了父类方法
        y_pred = self.linear(X) #调用linear的call方法，call方法中调用了自己的forward函数
        return y_pred

def train():
    x_data = torch.tensor([[1.0],[2.0],[3.0]])
    y_data = torch.tensor([[2.0],[4.0],[6.0]])

    model = LinearModel()
    criterion = torch.nn.MSELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #优化器，model.parameters()指对前面model的权值进行梯度下降，学习率是0.1
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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


if __name__ == '__main__':
    train()