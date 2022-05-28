import torch
import os
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms


 
#超参数设置
#DEVICE=torch.device('cuda'if torch.cuda.is_available() else 'cpu')          #转gpu
DEVICE=torch.device( 'cpu')          #转cpu
print(DEVICE)
EPOCH=2
BATCH_SIZE=256

# 创建 Inception 结构函数（模板）
class Inception(nn.Module):
    # 参数为 Inception 每个卷积核的数量,其它参数GoogLeNet结构图中都给定了
    #ch1x1:图中左数第一个卷积核的数量，ch3x3red:图中左数第二个进入3x3前的1x1降维用的卷积核的数量，以此类推
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        # 四个并联结构
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels,pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]      
        return torch.cat(outputs, 1)                                        #直接拼到一起,尺寸大小不变，维度=ch1x1+ch5x5+ch3x3+pool_proj




 # 创建辅助分类器结构函数（模板）
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        #self.avgPool = nn.AvgPool2d(kernel_size=5, stride=3)               #论文原文如此，但我输入分辨率并非224x244，需要修改  
        self.avgPool = nn.AvgPool2d(kernel_size=2, stride=1)                #（5-2）+1=4   
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)                                
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        #原文： aux1: N x 512 x 14 x 14   aux2: N x 528 x 14 x 14（输入）
        #print(x.size())
        x = self.avgPool(x)
        #print(x.size())
        # 原文:aux1: N x 512 x 4 x 4  aux2: N x 528 x 4 x 4（输出） 4 = (14 - 5)/3 + 1
        x = self.conv(x)
        x = torch.flatten(x, 1)     # 展平
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return x


# 创建卷积层函数（模板）
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
#网络模型构建GoogLeNet
#注意：GoogLeNet连接层参数与原版参数不一样,因为输入为81x81,注释内为输入为65情况下各层输出大小的计算
#另外，预测时未按照原论文将张量过一下softmax，后续有空加入
#本人因为显存不够，无法完整训练，仅仅确保了能够训练
class GoogLeNet(nn.Module):
    # aux_logits: 是否使用辅助分类器（训练的时候为True, 验证的时候为False)
    def __init__(self, num_classes=2, aux_logits=True, init_weight=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits                                                #设置辅助分类器为True
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)         #(81-7+6)/2+1=41
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)       # 当结构为小数时，ceil_mode=True向上取整，=False向下取整 
       #(41-3)/2+1=20
       # nn.LocalResponseNorm （此处省略）
        self.conv2 = nn.Sequential(
            BasicConv2d(64, 64, kernel_size=1),                                     #20
            BasicConv2d(64, 192, kernel_size=3, padding=1)                          #20
        )
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)                   #(20-3)/2+1=10

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)                  #9
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)                #9
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)                   #(10-3)/2+1=5

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)                 #5x5x(192+208+48+64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)                #5x5x(528)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)                   #(5-2)/2+1=3

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:      # 使用辅助分类器
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))                                 #输出特征图大小1x1
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        if init_weight:
            self._initialize_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x =self.maxpool3(x)

        x =self.inception4a(x)
        #print(x.size())
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        x = self.inception4e(x)
        x =self.maxpool4(x)
        #print(x.size())

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_logits:
            return x, aux1, aux2
        return x


    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)










 
#数据预处理
 
#数据路径
# aim_dir0=r'shdj'
# aim_dir1=r'sdj'
# source_path0=r'efeeeeee'
# source_path1=r'ddddddddd'
 
#数据增强
# def DataEnhance(sourth_path,aim_dir,size):
#     name=0
#     #得到源文件的文件夹
#     file_list=os.listdir(sourth_path)
#     #创建目标文件的文件夹
#     if not os.path.exists(aim_dir):
#         os.mkdir(aim_dir)
#
#     for i in file_list:
#         img=Image.open('%s\%s'%(sourth_path,i))
#         print(img.size)
#
#         name+=1
#         transform1=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.ToPILImage(),
#             transforms.Resize(size),
#         ])
#         img1=transform1(img)
#         img1.save('%s/%s'%(aim_dir,name))
#
#         name+=1
#         transform2=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.ToPILImage(),
#             transforms.ColorJitter(brightness=0.5,contrast=0.5,hue=0.5)
#         ])
#         img2 = transform1(img)
#         img2.save('%s/%s' % (aim_dir, name))
#
#         name+=1
#         transform3=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.ToPILImage(),
#             transforms.RandomCrop(227,pad_if_needed=True),
#             transforms.Resize(size)
#         ])
#         img3 = transform1(img)
#         img3.save('%s/%s' % (aim_dir, name))
#
#         name+=1
#         transform4=transforms.Compose([
#             transforms.Compose(),
#             transforms.ToPILImage(),
#             transforms.RandomRotation(60),
#             transforms.Resize(size),
#         ])
#         img4 = transform1(img)
#         img4.save('%s/%s' % (aim_dir, name))
#
#
# DataEnhance(source_path0,aim_dir0,size)
# DataEnhance(source_path1,aim_dir1,size)
 
#对文件区分为训练集，测试集，验证集
 
#归一化处理
normalize=transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
 
#训练集
path_1=r'D:\imagedb\imageC&D\train_0'
trans_1=transforms.Compose([
    transforms.Resize((81,81)),
    transforms.ToTensor(),
    normalize,
])
 
#数据集
train_set=ImageFolder(root=path_1,transform=trans_1)
#数据加载器
train_loader=torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE,
                                         shuffle=True,num_workers=0)
print(train_set.classes)
 
#测试集
path_2=r'D:\imagedb\imageC&D\train_0'
trans_2=transforms.Compose([
    transforms.Resize((81,81)),
    transforms.ToTensor(),
    normalize,
])
test_data=ImageFolder(root=path_2,transform=trans_2)
test_loader=torch.utils.data.DataLoader(test_data,batch_size=BATCH_SIZE,
                                        shuffle=True,num_workers=0)
 
#验证集
path_3=r'.D:\imagedb\imageC&D\train_0'
valid_data=ImageFolder(root=path_2,transform=trans_2)
valid_loader=torch.utils.data.DataLoader(valid_data,batch_size=BATCH_SIZE,
                                         shuffle=True,num_workers=0)
 
#定义模型
model=GoogLeNet().to(DEVICE)
#优化器的选择
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.0005)
 
 
#训练过程
def train_model(model,device,train_loader,optimizer,epoch):
    train_loss=0
    model.train()
    for batch_index,(data,label) in enumerate(train_loader):
        data,label=data.to(device),label.to(device)
        optimizer.zero_grad()
        output,auxO1,auxO2=model(data)
        loss=F.cross_entropy(output,label)
        lossAux1=F.cross_entropy(auxO1,label)
        lossAux2=F.cross_entropy(auxO2,label)
        loss = loss + lossAux1 * 0.3 + lossAux2 * 0.3
        loss.backward()
        optimizer.step()
        if batch_index%300==0:
            train_loss=loss.item()
            print('Train Epoch:{}\ttrain loss:{:.6f}'.format(epoch,loss.item()))
 
    return  train_loss
 
 
#测试部分的函数
def test_model(model,device,test_loader):
    model.eval()
    correct=0.0
    test_loss=0.0
 
    #不需要梯度的记录
    with torch.no_grad():
        for data,label in test_loader:
            data,label=data.to(device),label.to(device)
            output=model(data)
            test_loss+=F.cross_entropy(output,label).item()
            output = torch.softmax(output, dim=1)                                           #softmax
            pred=output.argmax(dim=1)
            correct+=pred.eq(label.view_as(pred)).sum().item()
        test_loss/=len(test_loader.dataset)
        print('Test_average_loss:{:.4f},Accuracy:{:3f}\n'.format(
            test_loss,100*correct/len(test_loader.dataset)
        ))
        acc=100*correct/len(test_loader.dataset)
 
        return test_loss,acc
 
 
#训练开始
list=[]
Train_Loss_list=[]
Valid_Loss_list=[]
Valid_Accuracy_list=[]
 
#Epoc的调用
for epoch in range(1,EPOCH+1):
    #训练集训练
    train_loss=train_model(model,DEVICE,train_loader,optimizer,epoch)
    Train_Loss_list.append(train_loss)
    torch.save(model,r'.\model%s.pth'%epoch)
 
    #验证集进行验证
    test_loss,acc=test_model(model,DEVICE,valid_loader)
    Valid_Loss_list.append(test_loss)
    Valid_Accuracy_list.append(acc)
    list.append(test_loss)
 
#验证集的test_loss
 
min_num=min(list)
min_index=list.index(min_num)
 
print('model%s'%(min_index+1))
print('验证集最高准确率： ')
print('{}'.format(Valid_Accuracy_list[min_index]))
 
#取最好的进入测试集进行测试
model=torch.load(r'.\model%s.pth'%(min_index+1))
model.eval()
 
accuracy=test_model(model,DEVICE,test_loader)
print('测试集准确率')
print('{}%'.format(accuracy))
 
 
#绘图
#字体设置，字符显示
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
 
#坐标轴变量含义
x1=range(0,EPOCH)
y1=Train_Loss_list
y2=Valid_Loss_list
y3=Valid_Accuracy_list
 
#图表位置
plt.subplot(221)
#线条
plt.plot(x1,y1,'-o')
#坐标轴批注
plt.ylabel('训练集损失')
plt.xlabel('轮数')
 
plt.subplot(222)
plt.plot(x1,y2,'-o')
plt.ylabel('验证集损失')
plt.xlabel('轮数')
 
plt.subplot(212)
plt.plot(x1,y3,'-o')
plt.ylabel('验证集准确率')
plt.xlabel('轮数')
 
#显示
plt.show()
