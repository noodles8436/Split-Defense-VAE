from __future__ import print_function

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# NOTE: 아래는 MNIST 데이터셋을 내려받을 때 "User-agent" 관련한 제한을 푸는 코드입니다.
#       더 자세한 내용은 https://github.com/pytorch/vision/issues/3497 을 참고해주세요.
from six.moves import urllib

import VQ_VAE

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

epsilons = [0, .02, .04, .06, .08, 0.1, 0.12, 0.14,
            0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3]

pretrained_model = "./classifier_model/lenet_mnist_model.pth"
NoSplitted_VAE_Dir = './NoSplitted_VAE/VQ-VAE.pth'
Splitted_VAE_12_Dir = './Splitted_VAE/VQ-VAE.pth'
use_cuda = True

_VAE = "VAE"  # VAE or Split_VAE
_VAE_Dir = NoSplitted_VAE_Dir
VQ_VAE.img_size = (24, 24)


split_size = VQ_VAE.img_size[0]
if _VAE_Dir is not None:
    VAE_Model = VQ_VAE.getModel()
    VAE_Model.load_state_dict(torch.load(_VAE_Dir, map_location='cpu'))
    VAE_Model.eval()
    pytorch_total_params = sum(p.numel() for p in VAE_Model.parameters())
    print(_VAE, pytorch_total_params)

# LeNet 모델 정의
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6400, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# MNIST 테스트 데이터셋과 데이터로더 선언
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Resize(size=(24, 24)),
    ])),
    batch_size=1, shuffle=True)

# 어떤 디바이스를 사용할지 정의
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# 모델 초기화하기
model = Net().to(device)

# 미리 학습된 모델 읽어오기
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# 모델을 평가 모드로 설정하기. 드롭아웃 레이어들을 위해 사용됨
model.eval()


# FGSM 공격 코드
def fgsm_attack(image, epsilon, data_grad):
    # data_grad 의 요소별 부호 값을 얻어옵니다
    sign_data_grad = data_grad.sign()
    # 입력 이미지의 각 픽셀에 sign_data_grad 를 적용해 작은 변화가 적용된 이미지를 생성합니다
    perturbed_image = image + epsilon * sign_data_grad
    # 값 범위를 [0,1]로 유지하기 위해 자르기(clipping)를 추가합니다
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 작은 변화가 적용된 이미지를 리턴합니다
    return perturbed_image


def test(model, device, test_loader, epsilon):
    # 정확도 카운터
    correct = 0
    adv_examples = []

    # 테스트 셋의 모든 예제에 대해 루프를 돕니다
    for data, target in test_loader:

        # 디바이스(CPU or GPU) 에 데이터와 라벨 값을 보냅니다
        data, target = data.to(device), target.to(device)

        # 텐서의 속성 중 requires_grad 를 설정합니다. 공격에서 중요한 부분입니다
        data.requires_grad = True

        # 데이터를 모델에 통과시킵니다
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # 로그 확률의 최대값을 가지는 인덱스를 얻습니다

        # 만약 초기 예측이 틀리면, 공격하지 않도록 하고 계속 진행합니다
        if init_pred.item() != target.item():
            continue

        # 손실을 계산합니다
        loss = F.nll_loss(output, target)

        # 모델의 변화도들을 전부 0으로 설정합니다
        model.zero_grad()

        # 후방 전달을 통해 모델의 변화도를 계산합니다
        loss.backward()

        # 변화도 값을 모읍니다
        data_grad = data.grad.data

        # FGSM 공격을 호출합니다
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        perturbed_data = defense(perturbed_data)

        # 작은 변화가 적용된 이미지에 대해 재분류합니다
        output = model(perturbed_data)

        # 올바른지 확인합니다
        final_pred = output.max(1, keepdim=True)[1]  # 로그 확률의 최대값을 가지는 인덱스를 얻습니다
        if final_pred.item() == target.item():
            correct += 1
            # 0 엡실론 예제에 대해서 저장합니다
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # 추후 시각화를 위하 다른 예제들을 저장합니다
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # 해당 엡실론에서의 최종 정확도를 계산합니다
    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # 정확도와 적대적 예제를 리턴합니다
    return final_acc, adv_examples


def defense(data):
    if _VAE is None:
        return data

    # print(data.size(), 'input')
    data = data.detach().squeeze().unsqueeze(0)
    # plt.imshow(data.cpu().squeeze().numpy(), cmap='gray')
    # plt.show()

    with torch.no_grad():
        if _VAE == "Split_VAE":
            split_img = data.unfold(1, split_size, split_size).unfold(2, split_size, split_size)
            split_img = split_img.contiguous().view(-1, split_img.size(0), split_size, split_size)
            # print(split_img.size())
            for i in range(len(split_img)):
                split_img[i], _, _, _ = VAE_Model(split_img[i].unsqueeze(0))

            split_axis_range = int(math.sqrt(len(split_img)))

            '''
            plt.imshow(split_img[0].cpu().squeeze().numpy(), cmap='gray')
            plt.show()
            plt.imshow(split_img[1].cpu().squeeze().numpy(), cmap='gray')
            plt.show()
            plt.imshow(split_img[2].cpu().squeeze().numpy(), cmap='gray')
            plt.show()
            plt.imshow(split_img[3].cpu().squeeze().numpy(), cmap='gray')
            plt.show()
            '''
            _stack_list = list()

            for _y in range(split_axis_range):
                _stack_list.append(torch.cat(
                    tuple(split_img[_y * 2:_y * 2 + split_axis_range].squeeze()), dim=1))

            x_hat = torch.cat(tuple(_stack_list[:]), dim=0).unsqueeze(0).unsqueeze(0)

            # print(x_hat.size())
            # plt.imshow(x_hat.cpu().squeeze().numpy(), cmap='gray')
            # plt.show()

        else:
            # print('data', data.size())
            x_hat, _, _, _ = VAE_Model(data.unsqueeze(0))
            # print('x_hat', x_hat.size())

        return x_hat


def test_Defense():
    accuracies = []
    examples = []

    # 각 엡실론에 대해 테스트 함수를 실행합니다
    for eps in epsilons:
        acc, ex = test(model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)

    # 각 엡실론에서 적대적 샘플의 몇 가지 예를 도식화합니다
    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()

    return accuracies


def get_acc_list():
    if os.path.exists("./acc.txt") is False:
        return

    acc_list = list()
    _file = open("./acc.txt", 'r')
    while True:
        _acc = list()
        values = _file.readline().strip("[""]").split(",")
        for i in values:
            if i == "":
                continue
            _acc.append(float(i.strip()))
        if len(_acc) == 0:
            break
        acc_list.append(_acc)
    _file.close()
    return acc_list


if __name__ == "__main__":
    acc_list = get_acc_list()

    '''
    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, acc_list[0], "*-", color='blue', label='default')
    plt.plot(epsilons, acc_list[1], "*-", color='green', label='VAE')
    plt.plot(epsilons, acc_list[2], "*-", color='red', label='4-Split-VAE')
    
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    
    plt.legend(loc='lower left', ncol=1, fontsize=16)
    plt.tight_layout()
    plt.savefig('result', bbox_inches='tight')
    plt.show()
    '''

    acc_diff = list()
    print(acc_list[2])
    print(acc_list[1])
    for i in range(len(acc_list[0])):
        acc_diff.append((acc_list[2][i] - acc_list[1][i]) * 100)

    print(acc_diff)