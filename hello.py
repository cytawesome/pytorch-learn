import torch
import torchvision

def forward(x, w):
    return x * w

def loss(x, y, w):
    y_pred = forward(x, w)
    return (y_pred - y) ** 2

if __name__ == '__main__':
    x_data = [1.0, 2.0, 3.0]
    y_data = [2.0, 4.0, 6.0]

    w = torch.tensor([1.0])
    w.requires_grad = True

    print("predict (before training)", 4, forward(4, w).item())
    for epcho in range(10):
        for x, y in zip(x_data, y_data): #zip函数使x_data,y_data一一配对，组成[(x,y), ...]的形式
            l = loss(x, y, w)
            l.backward()  #反向传播之后，之前的计算图释放，重新绘图
            print("\tgrad:", x, y, w.item(), w.grad.item())
            w.data = w.data - 0.1 * w.grad.data #w.grad也是tensor，用其操作就是在构建计算图，取data计算不会重新构建计算图
            #这里没必要，w.data也是个tensor，这不是in-place operation，会创建一个新的tensor赋给w.data，虽然对结果没影响，但就算python会回收原来的tensor，但感觉还是会造成内存开销，直接用w.data -= 0.1*grad即可
            #w.data -= 0.1 * w.grad.data 不能直接w -= 0.1*grad,我感觉结果不影响，但编译器会报错，因为in-place operation操作requires_grad=True的tensor可能会导致梯度计算出错，所以作者限制了这个操作。
            # 用w.data或detach等手段更改w的值才行，w.data已经被废弃了，但任然可用。原理是w.data这个tensor是requires_grad=False的，所以可以随意操作，但其操作w值框架检测不到，可能导致结果错误，所以不推荐，

            w.grad.data.zero_() #梯度数据清0
        print("progress:", epcho, l.item())
    print("predict (after training)", 4, forward(4, w).item())