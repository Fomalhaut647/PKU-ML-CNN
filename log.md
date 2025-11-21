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
