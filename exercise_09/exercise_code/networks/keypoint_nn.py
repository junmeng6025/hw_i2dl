"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

# TODO: Choose from either model and uncomment that line
# class KeypointModel(nn.Module):
class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        NOTE: You could either choose between pytorch or pytorch lightning, 
            by switching the class name line.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        # self.hparams.update(hparams)
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        # self.hparams = hparams
        self.batch_size = hparams["batch_size"]
        self.loss_fn = nn.MSELoss()

        in_h = 96
        in_w = 96
        out_h, out_w = self.calc_out_conv_layers(in_h, in_w, 3, 1)  # the first ConvLayer
        out_h, out_w = self.calc_out_conv_layers(out_h, out_w, 2, 2)  # Pooling layer

        modules = []
        for _ in range(self.hparams['num_layers']):
            modules.append(nn.Conv2d(32, 32, kernel_size=3, stride=1))
            out_h, out_w = self.calc_out_conv_layers(out_h, out_w, 3, 1)  # after ConvLayer
            modules.append(nn.BatchNorm2d(32))
            modules.append(nn.ReLU())
            modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
            out_h, out_w = self.calc_out_conv_layers(out_h, out_w, 2, 2)

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *modules,
            nn.Flatten(),
            nn.Linear(32*out_h*out_w, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 30)

        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def calc_out_conv_layers(self, in_h, in_w, ker, stri=1, pad=0):
        out_h = in_h
        out_w = in_w
        out_h = (out_h + 2*pad - ker)//stri + 1
        out_w = (out_w + 2*pad - ker)//stri + 1

        return out_h, out_w

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################

        # should return a tensor of size
        x = self.model(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x

    def general_step(self, batch, batch_idx, mode):
        images, keypoints = batch['image'], batch['keypoints']
        pred_keypoints = torch.squeeze(self.forward(images)).view(batch['image'].size(0), 15, 2)
        loss = self.loss_fn(pred_keypoints, keypoints)
        return loss/batch['image'].size(0)

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, 'train')
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, 'val')
        return {'val_loss': loss}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"])
        # optim = torch.optim.SGD(self.model.parameters(), lr=self.hparams["lr"], momentum=0.9, weight_decay=1e-4)
        return optim

class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
