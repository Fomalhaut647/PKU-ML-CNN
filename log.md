# 原始模型

Best Val Acc: 0.0916

~~~python
epochs = 10
batch_size = 32
lr = 1e-3

transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
~~~

![](./results/1.png)



# 4 层 CNN

Best Val Acc: 0.0034

~~~python
epochs = 10
batch_size = 32
lr = 1e-3

transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
~~~

![](./results/2.png)



# 降低 lr & 增大 batch_size

Best Val Acc: 0.1383

~~~python
epochs = 100
batch_size = 256
lr = 1e-4

transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
~~~

![](./results/3.png)
