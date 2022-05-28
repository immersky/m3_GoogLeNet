论文原文https://arxiv.org/pdf/1409.4842.pdf

代码参考了网上流传的博客，但是原始出处没有找到。

使用的数据集下载:https://pan.baidu.com/s/1rPZzQTE00r8lnc9Ott9j2Q?pwd=n5im




GoogLeNet相比VGG与AlexNet更加有创新，因此不能直接贴出结构就立马复现，必须先了解其特性，当然，第一次见到GoogLeNet时需要先看结构。

# 1.特点

实力有限，理解不到位处请见谅，望指正。

## 1.1引入了inception模块结构，如同下图所示

![20210608142304769](https://user-images.githubusercontent.com/74494790/170828132-85d28988-6f3e-45fe-b1e5-f572a3367dcd.png)
![20210608142304800](https://user-images.githubusercontent.com/74494790/170828138-61485f5e-d0b6-4b77-bedb-7eaa9bb3f8d7.png)

个人理解：previous layer送来了上一层的多维张量，随后在incecption模块中，分为4个copy并行（这里并行是从网络结构来谈的，而不是计算机运算）被上图中对应节点处理,随后直接合并为一个大张量(Filter concatenation)，比如3x3x9 和3x3x10合并为了3x3x19，而为了能够合并，要注意图中的1x1,3x3,5x5等卷积操作必须设置合适的步长与padding确保输出的前两维相同。其次，这样的“并联“结构没有把复杂的网络并联，而是将卷积和池化操作这种卷积神经网络的基本操作并联，感觉像是让网络在训练时自行挑选每一次最需要的操作。

## 1.2 最末端全连接层前使用平均池化

网络最后采用了average pooling（平均池化）来代替全连接层，该想法来自NIN（Network in Network），事实证明这样可以将准确率提高0.6%,最大池化往往更容易把图像中的”边缘“特征筛选出，而平均池化更考虑到背景信息



## 1.3 使用1x1卷积核降维

这里参考https://blog.csdn.net/qq_42308217/article/details/110350914

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020113011051574.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMzA4MjE3,size_16,color_FFFFFF,t_70)

假设现在有一个特征矩阵是512，如果不使用1×1的卷积核，直接使用64个5×5的卷积核进行卷积的话，那么总参数为5×5×512×64=819200，如果使用24个1×1卷积核进行降维，再使用64个5×5的卷积核进行卷积，则所需要的参数就会变为50688

## 1.4 梯度消失处理

图像过深往往引起梯度消失，梯度爆炸的问题，GoogLeNet网络结构挺深，但是使用了辅助分类器解决问题（详见下面2.0部分的网络结构图）

网络训练过程中，inception模块连接处直接分出一层，加上全连接输出到softmax进行分类！然后根据loss向前面网络的更新权重。

网络用于预测时，不使用辅助分类器，辅助分类器进行分类只是为了在训练时更新前面的网络的权重。

# 2.网络结构


![constructure](https://user-images.githubusercontent.com/74494790/170828146-aa9213d7-7d47-4718-a8d1-4cbdf0e56f8d.jpg)


![20210608142111804](https://user-images.githubusercontent.com/74494790/170828148-e0da3b87-a92c-4e6c-ad89-7ea017a64d6c.png)


