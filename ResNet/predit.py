import os
import json
import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from model import resnet50


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    img_path = "./ResNet/dataset/flower_data/sunflowers.jpg"
    assert os.path.exists(img_path), "File: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    json_path = "./ResNet/class_indices.json"
    assert os.path.exists(json_path), "File: '{}' does not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model = resnet50(num_classes=5).to(device)

    weight_path = "./ResNet/resnet50.pth"
    assert os.path.exists(weight_path), "File: '{}' does not exist.".format(weight_path)
    model.load_state_dict(torch.load(weight_path, map_location=device))

    model.eval()
    with torch.no_grad():
        # 将图片通过model正向传播，得到输出，将输入进行压缩，将batch维度压缩掉，得到最终输出（out）
        output = torch.squeeze(model(img.to(device))).cpu()
        # 经过softmax处理后，就变成概率分布的形式了
        predict = torch.softmax(output, dim=0)
        # 通过argmax方法，得到概率最大的处所对应的索引
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}    prob: {:.3}".format(class_indict[str(predict_cla)],
                                                  predict[predict_cla].numpy())

    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}    prob: {:.3}".format(class_indict[str(i)],
                                                   predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
