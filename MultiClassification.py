import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd

class NeuralNetwork(torch.nn.Module): #输入tensor的shape是(N*1*28*28)，
    # 但神经网络linear层要求输入的数据为一维向量，所以实际输入网络的tensor应该是(N*784)
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(28*28, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, 10)

    def forward(self, x): #x.shape = (N, 1, 28, 28)
        x = x.view(-1, 28*28)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        return self.linear5(x)

def main(epoch_size, device):
    if device == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device("cpu")
    #1 Prepare Data
    batch_size = 64
    transform = transforms.Compose([
        transforms.ToTensor(), #转成c*w*h
        transforms.Normalize((0.1307,), (0.3081,)) #标准化，这个数据是通过Minist数据集算出来的
    ]) #把0-255像素值改为0-1，把图像从w*h*c改为c*w*h。c为通道值，通道就是颜色，比如rgb是三通道而灰度图像是单通道

    train_dataset = datasets.MNIST('./dataset/mnist/',
                                   train=True,
                                   transform=transform,
                                   download=True) #说明从dataset取数据会先经过transform对数据进行操作

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    test_dataset = datasets.MNIST('./dataset/mnist/',
                                   train=False,
                                   transform=transform,
                                   download=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    #2 Design model
    model = NeuralNetwork()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5) #利用冲量优化网络

    #3 Train
    def train(): #一个epcho的train
        accumulate_loss = 0.0 #每训练300次输出一个累积loss的平均值
        for batch_id, (input, target) in enumerate(train_loader, 0):
            input, target = input.to(device), target.to(device) #数据加载到cpu或者gpu内存上
            Y_pred = model(input)
            loss = criterion(Y_pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accumulate_loss += loss.item()
            if batch_id % 300 == 299:
                print("[%d, %5d] loss: %.3f" % (epoch, batch_id + 1, accumulate_loss/300))
                accumulate_loss = 0.0

    # 4 test
    def test():  #在测试集上验证模型效果
        correct = 0
        total = 0
        with torch.no_grad():
            for (input, target) in test_loader:
                input, target = input.to(device), target.to(device)
                outputs = model(input)
                _, predicted = torch.max(outputs, 1) #沿着第一个纬度（行）取max，_是最大值，predicted是下标
                total += target.size(0)
                correct += (predicted == target).sum().item()
        print("Accuracy of the network on test set: %d %%" % (100 * correct / total))

    #train and test
    for epoch in range(epoch_size):
        train()
        if epoch % (epoch_size//10) == 0 or epoch == epoch_size-1: #进行到1/10进行一次test
            print("--------epcho {} testing on test set--------".format(epoch))
            test()
            print("--------testing finished--------")

def test1(): #测试softmax
    criterion = torch.nn.CrossEntropyLoss() #softmax,直接将linear层输出的结果z，先e^z,再e^z/sum(e^z),再sum(-ylog(y_pred))计算loss了
    Y = torch.LongTensor([2,0,1]) #表示三个数据，对应类别2，类别0，类别1。实际上model中最后linear层应该有3个输出
    Y_pred1 = torch.tensor([[0.1, 0.2, 0.9],  #2
                            [1.1, 0.1, 0.2],  #0
                            [0.2, 2.1, 0.1]]) #1
    Y_pred2 = torch.tensor([[0.8, 0.2, 0.3],  # 2
                            [0.2, 0.3, 0.5],  # 0
                            [0.2, 0.2, 0.5]]) # 1
    #显然第一组数据更切合Y
    loss1 = criterion(Y_pred1, Y)
    loss2 = criterion(Y_pred2, Y)
    print("Batch Loss1 =", loss1.item(), "\nBatch Loss2 =", loss2.item())

def test2():
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]], requires_grad=True)
    print(x.data.data is x.data)  # 每一次.data都会在内存中创建出一个tensor对象，经过这个对象和x共享底层数据，但仍然包含别的属性占用内存空间，所以内存地址不一致。但在不在代码中包含.data就不会创建这个内存，若含有.data就会创建这个tensor用于脱离父tensor的计算图（不建议用）
    print(x.data.data.data is x.data)  # 输出: False
    print(x.data.data.data is x)
    with torch.no_grad():
        print(x[1, 2])

def test3():
    train_dataset = datasets.MNIST('./dataset/mnist/',
                                   train=True,
                                   download=True)  # 说明从dataset取数据会先经过transform对数据进行操作
    print(train_dataset[0])

#------------------Excercise9-2------------
class OttoDataset(Dataset):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, delimiter=',', quotechar='"')
        self.df['target'] = self.df['target'].replace(
            {'Class_1': 0, 'Class_2': 1, 'Class_3': 2, 'Class_4': 3, 'Class_5': 4, 'Class_6': 5,
             'Class_7': 6, 'Class_8': 7, 'Class_9': 8}
        ).astype('float32') #将表示类别的字符串处理成离散值
        numpy_data = self.df.to_numpy(dtype='float32') #转numpy
        self.features = torch.from_numpy(numpy_data[:, 1:-1])
        self.labels = torch.from_numpy(numpy_data[:, -1:])
        self.len = numpy_data.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.len

#该模型在经过1000epoch训练后，loss降到0.08-0.09，accuracy为96%，并可以继续训练下降loss。但是，没有区分train和test set，所以可能过拟合。
class OttoModel(torch.nn.Module): #用于Otto数据处理的全链接神经网络，输入tensor的shape是(N*93)，
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(93, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, 64)
        self.linear4 = torch.nn.Linear(64, 32)
        self.linear5 = torch.nn.Linear(32, 9)

    def forward(self, x): #x.shape = (N, 93)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        return self.linear5(x) #损失函数采用CrossEntropy，所以不需要softmax处理，直接输出线性层

def train_exercise9_2(epoch_size, device):
    # 0 Pre-define parameters
    if device == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device("cpu")
    batch_size = 4096
    # 1 Prepare data
    train_dataset = OttoDataset('./dataset/OttoGroupProduct/train.csv')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataset = OttoDataset('./dataset/OttoGroupProduct/train.csv')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 2 Prepare Model
    model = OttoModel()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()  # 多分类交叉熵损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 3 train
    def batch_train(epoch_num):  # 在一个epoch的测试
        accumulated_loss, num = 0.0, 0  # 每隔5次计算一次平均loss
        for batch_id, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # 让数据加载到相应的cpu或gpu内存上
            labels = labels.view(inputs.size(0)) #criterion接受的labels是要求一维的
            labels = labels.to(torch.int64) #nvidia需要labels为int，不是float，否则会报错
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)

            accumulated_loss += loss.item()
            num += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if num == 5:
                print("[%d, %5d] loss: %.3f" % (epoch_num+1, batch_id + 1, accumulated_loss / num))
                accumulated_loss, num = 0.0, 0

    # 4 test and evaluate
    def test():
        correct = 0  # 正确预测个数
        total = 0  # 样本总数
        with torch.no_grad():  # 不要构成计算图
            for (inputs, labels) in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.view(inputs.size(0))
                labels = labels.to(torch.int64) #nvidia需要labels为int，不是float，否则会报错
                y_pred = model(inputs)
                _, predicted = torch.max(y_pred, 1)  # 沿着第一个纬度（行）取max，_是最大值，predicted是下标
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            print("Accuracy of the network on test set: %d %%" % (100 * correct / total))

    # train and test
    for epoch in range(epoch_size):
        batch_train(epoch)
        if epoch % (epoch_size // 10) == 0 or epoch == epoch_size - 1:  # 进行到1/10进行一次test
            print("--------epcho {} testing on test set--------".format(epoch))
            test()
            print("--------testing finished--------")

#------------------Excercise9-2------------


if __name__ == '__main__':
    #main(40, "gpu")
    train_exercise9_2(1000, 'gpu')
    #模型在经过1000epoch训练后，loss降到0.08-0.09，accuracy为96%，并可以继续训练下降loss。因为我感觉全链接模型过于简单，继续训练没有意义，没有继续探索的意义