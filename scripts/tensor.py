import torch


a = torch.tensor([1.0, 2.0, 4.0, 8.0])
b = torch.tensor([1.0, 0.5, 0.25, 0.125])

print(a + b)
print(torch.sigmoid(a))

d1 = torch.tensor([[1, 0], [0, 1]])
m2 = torch.tensor([[1, 2], [3, 4]])  

print(m2 * d1)

m3 = torch.sigmoid(torch.tensor([-1.0, 10.0, 0, -10]))
print(m3)
print(torch.relu(m3))

