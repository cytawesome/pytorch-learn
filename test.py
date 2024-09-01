class cat:
    def __init__(self):
        self.name = "cat"

def test(x):
    print(id(x))
    x = cat()
    x.name = "x"
    print(id(x))

def test1():
    a = cat()
    a.name = "a"
    print(id(a))
    test(a)
    print(a.name)

if __name__ == '__main__':
    test1()