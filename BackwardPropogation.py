import torch

def exercise4_4():
    x_data = [1.0, 2.0, 3.0]
    y_data = [2.0, 4.0, 6.0]
    w1 = torch.tensor([1.0], requires_grad=True) #leaf tensor
    w2 = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([3.0], requires_grad=True)
    a = 0.01 #学习率

    for epcho in range(10000): #训练100次
        print("start epcho ", epcho)
        for x, y in zip(x_data, y_data): #采用每个数据都单独进行梯度下降
            y_pred = w1*x*x + w2*x + b #y_pred是intermediate tensor，grad为None，但grad_fn指明了是何种计算
            l = (y_pred - y)**2
            l.backward() #反向传播，释放计算图
            print(
                "\tgrad x:{}, y:{}, w1:{} {}, w2:{} {}, b:{} {}".format(
                    x, y, w1.item(), w1.grad.item(), w2.item(), w2.grad.item(), b.item(), b.grad.item()
                )
            )

            with torch.no_grad(): #说明这是in-operation操作，不会将下面计算添加到计算图里面去，要不然w1就变成intermediate tensor了
                # 更新权重，-=才是in-operation,w1 = w1 - a*grad会赋予w1一个新的tensor对象
                w1 -= a*w1.grad
                w2 -= a*w2.grad
                b -= a*b.grad
                # 梯度数据清0,要不然grad会累加
                w1.grad.data.zero_()
                w2.grad.data.zero_()
                b.grad.data.zero_()
        print("progress: epcho:{}, loss:{}".format(epcho, l.item()))
    print("predict (after training)", w1.item()*16+w2.item()*4+b.item())



if __name__ == '__main__':
    exercise4_4()