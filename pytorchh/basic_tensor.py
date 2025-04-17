import torch
print(torch.__version__)

if torch.cuda.is_available():
    print("Available")
else:
    print("Not available")


#creating tensor

a = torch.empty(2,3)

b = torch.zeros(2,3)

c = torch.ones(2,3)

torch.manual_seed(0)
random_tensor = torch.rand(2,3) #values between random 0 and 1

#custom tensor
custom = torch.tensor(data=[[1,2,3],[1,2,3],[1,2,4],[1,2,4]])
# print(custom)
# print(custom.shape)



#linspace
linear = torch.linspace(0,10,10)
# print(linear)
# print(linear.shape)

print(torch.eye(6))


#random values of some matrix and with range

random_0_1 = torch.rand(5,5)
# print(random_0_1)


print(torch.randn(3,3)) # bell curve

print(torch.randint(10,20,(3,3),dtype=torch.float16))

#if you want to copy some tensor shape
linear_cpy = torch.empty_like(linear)
print(linear_cpy)

linear_cpy.to(dtype=torch.float16)

