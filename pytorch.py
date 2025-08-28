from torch import nn
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import math

lista_imgs = []
if __name__ ==  '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 4


    x = datasets.CIFAR10(root = "./data", train = True, transform=transform, download=True)

    xl = DataLoader(x, batch_size = batch_size, shuffle=True, num_workers=2)

    teste = datasets.CIFAR10(root = "./data", train = False, transform = transform, download=True)

    testel = DataLoader(teste, batch_size = batch_size, shuffle=True, num_workers=2)
    
    def mostrar_imagem(tensor):
        tensor = tensor.squeeze(0)
        num_channels = tensor.size(0)
        cols = math.ceil(math.sqrt(num_channels))
        rows = math.ceil(num_channels / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        if rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        for i in range(num_channels):
            ax = axes[i]
            channel = tensor[i].detach().numpy()
            ax.imshow(channel)
            ax.set_title(f"Canal {i+1}")
            ax.axis("off")
        for j in range(num_channels, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.show()
            
        
    class model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
        def forward(self, x, teste):
            if(teste==True):
                lista_imgs.append(x)
            x = self.conv1(x)
            if(teste==True):
                lista_imgs.append(x)
            x = self.pool(nn.functional.relu(x))
            if(teste==True):
                lista_imgs.append(x)
            x = self.conv2(x)
            if(teste==True):
                lista_imgs.append(x)
            x = self.pool(nn.functional.relu(x))
            if(teste==True):
                lista_imgs.append(x)
            x = torch.flatten(x, 1)
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            y = self.fc3(x)
            return y
    '''
    epochs = 5
    lr = 0.001
    criterio = nn.CrossEntropyLoss()
    rede = model()
    rede.train()
    optimizer = torch.optim.SGD(rede.parameters(), lr = lr)
    #training
    
    for i in range(epochs):
        loss_epoca = 0.0
        num_batches = 0
        for batch, (x,y) in enumerate(xl):
            pred = rede(x, False)
            loss = criterio(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoca += loss.item()
            num_batches = num_batches + 1
        print(f"loss in epoch {i}: {loss_epoca/num_batches}")
    #evaluation
    
    '''
    PATH = './cifar_net.pth'
    #torch.save(rede.state_dict(), PATH)
    rede = model()
    rede.load_state_dict(torch.load(PATH, weights_only=True))

    imagem, label = teste[5]
    y = rede(imagem.unsqueeze(0), True)

    mostrar_imagem(lista_imgs[0])
    '''
    imagemtop = lista_imgs[1]
    imagem2 = torch.squeeze(imagemtop)
    print(lista_imgs[1].size())
    print(imagem2.size(0))
    plt.imshow(imagem2[1].detach().numpy(), cmap="gray")
    plt.title("dsf")
    plt.show()
    '''
            
            
    
    
        
    