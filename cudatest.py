import torch

print(torch.version.cuda)
a = torch.randn(3,3).cuda(0)
b = torch.randn(3,3).cuda(0)
print(a+b)

print(torch.linalg.inv(a)) #这句会报错，我们的GPU上求逆报错，可能是cuda版本问题啥的

#下面的代码把a复制到cpu就可以正常求逆
# x =  a.to('cpu')
# print(torch.linalg.inv(x))