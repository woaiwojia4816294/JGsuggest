import torch

pthfile = r'D:\Users\赵瑞妮\PycharmProjects\yunjey\tutorials\01-basics\feedforward_neural_network\model.ckpt'
net = torch.load(pthfile, map_location=torch.device('cpu'))

print(type(int))
print(len(net))

for k, v in net.items():
    print(k, v)

for key in net.keys():
    print(key)    # fc1.weight
                  # fc1.bias
                  # fc2.weight
                  # fc2.bias
