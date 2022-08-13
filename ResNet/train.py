import os
import sys
import json
import torch
import torch.utils.data
from torch import nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from model import resnet50


def main():
    # 判断是否有使用 GPU 的条件
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))

    # 定义训练集和验证集的数据预处理方式
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(224),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])}
    # 根目录
    data_root = "..\ResNet"
    # 数据集目录
    image_path = os.path.join(data_root, "dataset", "flower_data")
    # 判断数据集目录是否存在
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # 对训练集进行处理
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    # 显示类别名及对应的标签
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for val, key in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open("class_indices.json", "w") as json_file:
        json_file.write(json_str)

    batch_size = 16
    # 计算使用num_workers的数量
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print("Using {} dataloader workers every process.".format(nw))

    # 构建dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=nw)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)

    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=nw)
    print("Using {} images for training, {} images for validation.".format(train_num, val_num))

    # 实例化网络
    net = resnet50()
    # 加载预训练模型
    model_weight_path = "../ResNet/resnet50-pre.pth"
    assert os.path.exists(model_weight_path), "File {} does not xeist.".format(model_weight_path)
    # 在gpu上训练好的参数，若在cpu的机器上使用参数，加载时需要添加map_location
    net.load_state_dict(torch.load(model_weight_path, map_location="cpu"))

    # 修改最后的全连接层
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)
    net.to(device)

    # 定义交叉熵损失函数
    loss_function = nn.CrossEntropyLoss()

    # 定义优化器
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 3
    best_acc = 0.0
    save_path = "./resnet50.pth"
    train_steps = len(train_loader)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar_desc = "train epoch[{} / {}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "Valid epoch [{} / {}]".format(epoch + 1, epochs)

        val_accurate = acc / val_num
        print("[epoch %d] train_loss: %.3f val_accuracy: %.3f" %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print("Finish training")


if __name__ == '__main__':
    main()

