from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm import tqdm
import time


class VAE_MNIST_Dataset(Dataset):
    def __init__(self, split_size=14, isTrain=True):
        train_data = datasets.MNIST(root='./data', train=isTrain, download=True,
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                  transforms.Resize(size=(24, 24)),]))
        w = 28  # MNIST
        h = 28  # MNIST

        print('VAE-MNIST-DATASET Preparing..')
        print('is Train : ', str(isTrain))
        print('Dataset Size : ', len(train_data))
        print('')

        time.sleep(1)

        my_x = list()
        my_y = list()
        for i in tqdm(range(len(train_data))):
            img, label = train_data[i]
            split_img = img.unfold(1, split_size, split_size).unfold(2, split_size, split_size)
            split_img = split_img.contiguous().view(-1, split_img.size(0), split_size, split_size)

            for i in range(len(split_img)):
                my_x.append(split_img[i])
                my_y.append(int(int(label) * 4 + i))

        print('')
        print('VAE-MNIST-DATA Prepared!')
        print('')

        self.x_data = my_x
        self.y_data = my_y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)
