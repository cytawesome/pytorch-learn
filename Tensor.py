import torch

def test1():
    w = torch.tensor([1.], requires_grad = True)
    y = w + 1
    l = (y-2)**2
    l.backward()
    print(w.grad.item())

def test2():
    x = torch.tensor(100., requires_grad = True)

    print(x.data)
    print(x.data.requires_grad)

    y = 2*x*x
    y.backward()
    with torch.no_grad():
        #x = x *  100 这不是in-place operation, 这样就会创建一个新的x且requires_grad = False，在循环中接下来就构不成计算图了
        x *= 100

    #y.backward()
    print(x)
    print(x.grad)

def test3():
    w = torch.tensor([1.], requires_grad = True)
    initial_addr = w.data.data_ptr()
    w.data = w.data - 0.1
    print(w.data.data_ptr() == initial_addr)
    print(w.item())

def test4():
    x = torch.tensor(100., requires_grad=True)

    print(x.data)
    print(x.data.requires_grad)

    y = 2 * x * x
    #y.backward()
    b = x.detach() #detach的作用就是创建一个新的Tensor对象但requires_grad=False，同时与原对象共享data数据。因为False，所以不会进行图计算。但不难在x被backward之前修改，这样会报错。
    b *= 100

    y.backward()
    print(x, b)
    print(x.grad)

if __name__ == '__main__':
    test2()