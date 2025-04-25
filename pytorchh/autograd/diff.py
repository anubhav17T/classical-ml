import torch

x = torch.tensor(data=4.0, requires_grad=True)
y = x ** 2
print(y, x)
y.backward()
print(x.grad)

x = torch.tensor(data=4.0, requires_grad=True)
y = x ** 2
z = torch.sin(y)

z.backward()
print(x.grad)

w = torch.tensor(data=1.0, requires_grad=True)
x = torch.tensor(data=5.0, requires_grad=True)
b = torch.tensor(data=0.0, requires_grad=True)

z = w * x + b
print(z)
y_predictive = torch.sigmoid(z)


def binary_cross_entropy_loss(prediction, target):
    epsilon = 1e-8  # To prevent log(0)
    prediction = torch.clamp(prediction, epsilon, 1 - epsilon)
    return -(target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction))


loss = binary_cross_entropy_loss(target=0, prediction=y_predictive)
print(loss)
loss.backward()
print(w.grad)
print(b.grad)

arr = [0,1,0,3,12]

pos = 0
for i in range(0, len(arr)):
    if arr[i] != 0:
        arr[pos] = arr[i]
        pos += 1

end = len(arr)-1
while end >=pos:
    arr[end] = 0
    end-=1
print(arr)



x = [1,2,3]
n = len(x)
mat = []
for i in range(n):
    mat.append([0]*n)

for i in range(0,len(mat)):
    for k in range(0,len(mat)):
        if i == k:
            mat[i][k] = x[i]
print(mat)



