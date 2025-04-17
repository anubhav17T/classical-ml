#Scalar operation
import torch

a = torch.rand(2,2)
print(a)

print(a+2)
print(a-2)
print(a/2)
print(a*3)


#element wise / item wise
a = torch.randint(0,10,(2,2),dtype=torch.float32)
b = torch.randint(0,10,(2,2),dtype=torch.float32)
print(a+b)
print(a-b)


c =  a+b
print(c)
print(torch.sum(c)) #sum of a matrix
print(torch.sum(c,dim=0)) #sum of matrix by column
print(torch.sum(c,dim=1)) #sum of matrix by row

#similarily you can use max, min, mean , median on all tensors
torch.prod(c)
torch.std(c)
torch.var(c)
torch.max(c)
torch.min(c)
print(torch.argmax(c))  # it tells us the biggest element position

print(torch.argmin(c))  # it tells us the smalles element position





#matrix multiplication, dotproduct, crossproduct, transpose, inverse , determinant, inverse


a = torch.randint(0,10,(4,3))
b = torch.randint(0,5,(3,4))

print(torch.matmul(a,b))








