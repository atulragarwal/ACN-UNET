import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p = 0.5),
#             nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.conv(x)

# class AtulConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(AtulConv, self).__init__()
#         self.aConv = nn.Sequential(
#             # nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
#             # nn.ReLU(inplace = True),
#             # nn.Dropout(p=0.5),
#             # nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
#             # nn.ReLU(inplace = True),
#             # nn.MaxPool2d(2,2),
#             nn.Conv1d(in_channels, out_channels, 1, stride = 1, padding = 0),
#             # nn.BatchNorm2d(out_channels),
#             nn.Threshold(0,0,inplace = False),
#             nn.Conv1d(in_channels, out_channels, 1, stride = 1, padding = 0),
#             # nn.BatchNorm2d(out_channels),
#             nn.Threshold(0,0,inplace = False),


#         )
#     def forward(self, x):
#         return self.aConv(x)

class RohitConv(nn.Module):
    def __init__(
            self
    ):
        super(RohitConv, self).__init__()
        self.aConv = nn.Sequential(
            # nn.Threshold(254,0,inplace = True),
            # nn.Conv2d(3, 3, 1, stride=1, padding=0),
            # # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace = True),
            nn.Conv2d(3, 32, 3, 1, padding = 1),
            # nn.Conv1d(3, 3, 1, stride = 1, padding = 0),
            # nn.BatchNorm2d(out_channels),
            nn.Threshold(0.5,0,inplace = True),
            nn.Dropout(p = 0.5),
            nn.Conv2d(32, 64, 3, 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64,3,3,stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(3,1,3,stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            # nn.Linear(9216, 128),
            # nn.ReLU(inplace = True),
            # nn.Linear(128, 2),
            # nn.ReLU(inplace = True),

        )

    def forward(self, x):
        # x = x.squeeze()
        # x = x.transpose(0,1)
        x = self.aConv(x)
        return (x)

# def test():
#     x = torch.randn((3, 1, 161, 161))
#     model = UNET(in_channels=1, out_channels=1)
#     preds = model(x)
#     print(preds.shape)
#     print(x.shape)
#     assert preds.shape == x.shape

# if __name__ == "__main__":
#     test()