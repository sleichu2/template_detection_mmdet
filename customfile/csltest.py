class A(object):
    def __init__(self):
        print("A")
    def hello(self):
        print("helloA")#我是褚石磊
    def __call__(self, a):
        print(a)

class B(A):
    def __init__(self):
        super(B,self).__init__()
        print("B")
    def hello(self):
        print("helloB")

if __name__ == '__main__':
    b = B()
    b("hello")
