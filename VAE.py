import numpy as np
import os
from PIL import Image
import torch
import torchvision
import datetime

num_epochs = 20

def to_img(x):
    #x = 0.5 * (x + 1)
    x = x.clamp(0, 255)
    x = x.view(x.size(0), 3, 100, 100)
    return x

def load_dataset(data_path):
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        num_workers=0,
        shuffle=False
    )
    return train_loader

class autoencoder(torch.nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3,88, (2, 2)),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d((2,2)),
           
            torch.nn.Conv2d(88,173, (2, 2)),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d((2,2)),

            torch.nn.Conv2d(173,258, (2, 2)),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d((2,2)),

            torch.nn.Conv2d(258,343, (2, 2)),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d((2,2)),

            torch.nn.Conv2d(343,428, (2, 2)),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d((2,2)),

            torch.nn.Conv2d(428,512, (1, 1)),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d((2,2))
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512,482, (7, 7)),
            torch.nn.ReLU(True),
            
            torch.nn.ConvTranspose2d(482,452, (7, 7)),
            torch.nn.ReLU(True),
            
            torch.nn.ConvTranspose2d(452,422, (7, 7)),
            torch.nn.ReLU(True),
            
            torch.nn.ConvTranspose2d(422,392, (7, 7)),
            torch.nn.ReLU(True),
            
            torch.nn.ConvTranspose2d(392,362, (7, 7)),
            torch.nn.ReLU(True),
            
            torch.nn.ConvTranspose2d(362,332, (7, 7)),
            torch.nn.ReLU(True),
            
            torch.nn.ConvTranspose2d(332,302, (7, 7)),
            torch.nn.ReLU(True),
            
            torch.nn.ConvTranspose2d(302,272, (7, 7)),
            torch.nn.ReLU(True),
            
            torch.nn.ConvTranspose2d(272,242, (7, 7)),
            torch.nn.ReLU(True),
            
            torch.nn.ConvTranspose2d(242,212,(7, 7)),
            torch.nn.ReLU(True),
            
            torch.nn.ConvTranspose2d(212,182,(7, 7)),
            torch.nn.ReLU(True),
            
            torch.nn.ConvTranspose2d(182,152,  (7, 7)),
            torch.nn.ReLU(True),

	    torch.nn.ConvTranspose2d(152,122,(7,7)),
	    torch.nn.ReLU(True),

	    torch.nn.ConvTranspose2d(122,92,(7,7)),
	    torch.nn.ReLU(True),

	    torch.nn.ConvTranspose2d(92,62,(6,6)),
	    torch.nn.ReLU(True),

	    torch.nn.ConvTranspose2d(62,32,(6,6)),
	    torch.nn.ReLU(True),

	    torch.nn.ConvTranspose2d(32,3,(6,6)),
	    torch.nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        
        x = self.decoder(x)
   
        return x


def train():    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = autoencoder().to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-5)

    for epoch in range(num_epochs):
        print('Training of epoch [{}/{}]'.format(epoch+1, num_epochs))
        k = 0
        for data in load_dataset("."+os.sep+"test_training"+os.sep):
            print(datetime.datetime.now(), epoch+1, k)
            k = k + 1
            img, _ = data
            img = torch.autograd.Variable(img).to(device)
        # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
        # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data[0]))

        if epoch % 5 == 0:
            pic = to_img(output.cpu().data)
            torchvision.utils.save_image(pic, './dc_img/image_{}.png'.format(epoch))

        torch.save(model.state_dict(), './conv_autoencoder.pth')

train()


