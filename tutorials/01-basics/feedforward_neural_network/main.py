import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration  # pytorch支持gpu，可以通过to(device)函数将数据从内存转移到显存中
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = torch.cuda.is_available()
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# img = img.to(device)
# label = label.to(device)
# 对于模型也是，使用.to(device)或.cuda将网络放到GOU显存 model = Net()
# model.to(device) 使用序号为0的GPU
# model.to(device1) 使用序号为1的GPU

# Hyper-parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


# Fully connected neural network with one hidden layer  有一个隐藏层的全连接神经网络
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):  # 要传值的时候才这么写
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)
# 如果初始化的时候，有形参，实例化一个NeuralNet模型的时候，就要传实参

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
total_step = len(train_loader)  # 每一轮训练多少步
for epoch in range(num_epochs):  # 所有数据训练5轮
    for i, (images, labels) in enumerate(train_loader):  # 不管这个元组用什么表示，反正他都是这个意思
        # Move tensors to the configured device  本来读入的图片是 100个 1*28*28
        images = images.reshape(-1, 28*28).to(device)  # images(100,784),一次训练加载进来的图片是
        labels = labels.to(device)
        
        # Forward pass  前向传播
        outputs = model(images)  # 这么算出来的 outputs就是在 gpu 上的
        loss = criterion(outputs, labels)
        
        # Backward and optimize  反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:  # .format()括号中的内容填入前面的大括号中，现在运行到第几轮了 epoch，每一个 epoch 的多少步
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():  # with torch.no_grad() 或 @torch.no_grad()下面的数据不需要计算梯度，也不会进行反向传播
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)  # 为什么图片和标签都需要单独放在 gpu 上
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # correct += (predicetd == labels) 预测值和标签值相同时加上，correct本来是 0 ，+= 以后 correct = tensor([1,1,1,1,1],device='cuda:0',.dtype=torch.unit8)
        # correct.sum()  correct中的所有元素求和
        # .item() 的作用是 only one element tensors can be converted to python scalars
        # 如 将tensor(100, device='cuda:0') 变成 标量 100

    print('Accuracy of the network on the 10000 test images: {.4f} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model, 'wholeModel.pth')
torch.save(model.state_dict(), 'model.ckpt')