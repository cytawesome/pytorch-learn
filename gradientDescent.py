def f_x(w, x): #模型采用w*x
    return w*x

def loss(x, y, w):
    y_pred = f_x(w, x)
    return (y_pred - y)**2

def cost(x_data, y_data, w): #总的loss
    cost = 0
    for i in range(len(x_data)):
        loss_ =  loss(x_data[i], y_data[i], w)
        cost += loss_
    return cost/len(x_data)

def grad(x_data, y_data, w): #计算w位置，x_data,y_data为数据的cost的梯度
    grad = 0
    for i in range(len(x_data)):
        x, y = x_data[i], y_data[i]
        grad_i = 2*w*(w*x - y)
        grad += grad_i
    return grad/len(x_data) #计算梯度完成

def gradDesc(x_data, y_data, epcho): #梯度下降
    w = 100 #设一个初始值
    a = 0.0001 #学习率
    for i in range(epcho):
        w = w - a*grad(x_data, y_data, w)
        print("epcho:{}, w:{}, cost:{}".format(i+1, w, cost(x_data, y_data, w)))

def batchGradDesc(x_data, y_data, epcho): #分批梯度下降,避免陷入grad为0的位置
    batch_size = 10
    w, a = 10, 0.001
    batch_num = int(len(x_data) / batch_size)
    for turn in range(epcho):
        print("--------------------start epcho:{} calculating--------------------------".format(turn+1))
        for i in range(batch_size):
            start, end = i*batch_num, min((i+1)*batch_num, len(x_data))
            x_i = x_data[start:end]
            y_i = y_data[start:end]
            w = w - a*grad(x_i, y_i, w)
            print("epcho {} , start {}, end {}, w {}".format(turn+1, start, end, w))
        print("-----------------------------epcho:{} finish calculating, w is {}, cost is {}-----------------------".format(turn + 1, w, cost(x_data, y_data, w)))

if __name__ == '__main__':
    x_data = [2, 5, 12]
    y_data = [6, 15, 36]
    gradDesc(x_data, y_data, 10000)