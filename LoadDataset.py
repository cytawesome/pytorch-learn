from itertools import accumulate

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.sigmoid(self.linear3(x)) #最后这里输出y_pred不能是relu，因为relu在输入小于0时输出为0，这个在loss中log(0)会报错
        return x

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, -1:])

    def __getitem__(self, index): #__是魔法方法，说明该class支持下标访问，比如dataset[index]
        return self.x_data[index], self.y_data[index]

    def __len__(self): #len(dataset)执行的函数
        return self.len

#测试Dataloadder作用制作的建议数据集
class TestDataset(Dataset):
    def __init__(self, n):
        self.data = [i for i in range(n)]
        self.len = n

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

def train():
    dataset = DiabetesDataset('./dataset/diabetes.csv.gz')
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

    model = NeuralNetwork()
    criterion = torch.nn.BCELoss(size_average=True)  # 二元交叉熵
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epcho in range(100):
        for i, data in enumerate(train_loader, 0): #for i, (input, labels) 也行，enumerate(train_loader, 0)中的0表示索引从0开始
            #1 Prepare data
            inputs, labels = data #这里dataLoader会根据Dataset读取mini-batch数量的样本并转化为tensor矩阵
            #2 Forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epcho, i, loss.item())
            #3 Backward
            optimizer.zero_grad()
            loss.backward()
            #4 Update weighs
            optimizer.step()

def test():
    dataset = DiabetesDataset('./dataset/diabetes.csv.gz')
    print(dataset[10])

def test1():
    x = torch.tensor([[[1, 2, 3]],
                  [[4, 5, 6]],
                  [[7, 8, 9]]])
    print(x.view(1,1, -1))

def test_dataloader():
    dataset = TestDataset(32)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    #print(dataloader[2]) 报错，dataloader没有实现__getitem__()方法，不支持下标访问。但dataloader依然是个iterable对象，只要实现了__iter__()方法的就是可迭代对象，因为通过这个方法可以获得其迭代器
    #print(next(dataloader)) #报错，说明dataloader只是个可迭代对象，不是迭代器本身
    for data in dataloader:
        print(data, end=",")
    print()
    for data in dataloader: #The result of every traverse of dataloader is different when setting shuffle is true
        print(data, end=",")

#-----------------Excercise 8-1----------------
class TitanicDataset(Dataset): #针对titanic数据构造相应dataset
    def __init__(self, filepath):
        self.df = pd.read_csv('./dataset/titanic/train.csv', delimiter=',', quotechar='"')
        self.df['Sex'] = self.df['Sex'].replace({'male': 1, 'female': 0}).astype('float32') # 使用 replace 方法将 'male' 替换为 1，'female' 替换为 0
        self.df = self.df.dropna(axis=0, subset=['Survived','Pclass','Sex','Age','SibSp','Parch']) #删除feature列有NaN的数据行
        numpy_data = self.df[['Survived','Pclass','Sex','Age','SibSp','Parch']].to_numpy(dtype='float32') #人工筛选出'Survived','Pclass','Sex','Age','SibSp','Parch'作为feature，别的feature不太会影响生存
        self.features = torch.from_numpy(numpy_data[:, 1:])
        self.labels = torch.from_numpy(numpy_data[:, 0:1])
        #对age列归一化
        age_data = self.features[:, 2]
        min_age = age_data.min()
        max_age = age_data.max()
        normalized_data = (age_data - min_age) / (max_age - min_age) #归一化
        self.features[:,2] = normalized_data
        self.len = self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.len

class TitanicModel(torch.nn.Module): #定义用于训练titanic数据的神经网络,设计4层全链接神经网络
    def __init__(self): #输入数据有5个feature
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 20)
        self.linear2 = torch.nn.Linear(20, 40)
        self.linear3 = torch.nn.Linear(40, 20)
        self.linear4 = torch.nn.Linear(20, 10)
        self.linear5 = torch.nn.Linear(10, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x): #构造计算图
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.sigmoid(self.linear5(x)) #x输出0-1，代表船员存活的概率
        return x

def exercise8_1_train(epoch_size, device):
    #0 Pre-define parameters
    if device == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device("cpu")
    batch_size = 32
    #1 Prepare data
    train_dataset = TitanicDataset('./dataset/titanic/train.csv')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataset = TitanicDataset('./dataset/titanic/train.csv')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    #2 Prepare model
    model = TitanicModel()
    model.to(device)
    criterion = torch.nn.BCELoss(size_average=True) #二分类交叉熵损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    #3 train
    def batch_train(epoch_num): #在一个epoch的测试
        accumulated_loss, num = 0.0, 0 #每隔10次计算一次平均loss
        for batch_id, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device) #让数据加载到相应的cpu或gpu内存上
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)

            accumulated_loss += loss.item()
            num += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if num == 10:
                print("[%d, %5d] loss: %.3f" % (epoch_num, batch_id + 1, accumulated_loss/num))
                accumulated_loss, num = 0.0, 0


    #4 test and evaluate
    def test():
        correct = 0 #正确预测个数
        total = 0 #样本总数
        with torch.no_grad(): #不要构成计算图
            for (inputs, labels) in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                y_pred = model(inputs)
                for index in range(y_pred.size(0)):
                    if y_pred[index].item() > 0.5:
                        result = 1
                    else:
                        result = 0
                    if labels[index].item() == result:
                        correct += 1
                total += labels.size(0)
            print("Accuracy of the network on test set: %d %%" % (100 * correct / total))

    #train and test
    for epoch in range(epoch_size):
        batch_train(epoch)
        if epoch % (epoch_size // 10) == 0 or epoch == epoch_size - 1:  # 进行到1/10进行一次test
            print("--------epcho {} testing on test set--------".format(epoch))
            test()
            print("--------testing finished--------")


def exercise8_1_test():
    train_dataset = TitanicDataset('./dataset/titanic/train.csv')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)


#-----------------Excercise 8-1----------------

if __name__ == '__main__':
    #train()
    exercise8_1_train(200, 'gpu')