import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return torch.flatten(input, start_dim=1)

# CNN part of single task network - takes a spectrogram as input
class STN(nn.Module):
    out_size = 256
    
    def __init__(self):
        super().__init__()
        
        self.cnn = nn.Sequential(
                    
                    # layer 2
                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=32),
                    nn.ELU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(p=0.1),
                    
                    # layer 3
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                    nn.ELU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    # layer 4
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=64),
                    nn.ELU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(p=0.1),
                    
                    # layer 5
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                    nn.ELU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    # layer 6
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                    nn.ELU(),
                    
                    # layer 7
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
                    nn.BatchNorm2d(num_features=STN.out_size),
                    nn.ELU(),
                    
                    # global average pooling
                    nn.AdaptiveAvgPool2d((1, 1)),  # (N, 256, H, W) -> (N, 256, 1, 1)
                    Flatten(),  # (N, 256, 1, 1) -> (N, 256)
                    nn.BatchNorm1d(num_features=STN.out_size)
        
        )
    
    def forward(self, x):
        return self.cnn(x)

class MTN(nn.Module):
    def __init__(self, num_stns):  # num_targets is a list of targets for each STN here
        super().__init__()
        
        self.shared_block = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
                    nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
                    nn.ELU()
        )
        
        self.stns = nn.ModuleList(STN() for _ in range(num_stns))
    
    def forward(self, x):
        x = self.shared_block(x)
        return tuple(stn(x) for stn in self.stns)

# makes only the computations necessary to predict the final output (genre)
# given a spectrogram as input
class MTNFC(nn.Module):
    def __init__(self, mtn, stn_targets):
        super().__init__()
        
        self.mtn = mtn
        self.stn_targets = stn_targets
        self.fcls = nn.ModuleList([nn.Sequential(
                    
                    # fc
                    nn.Linear(in_features=STN.out_size, out_features=256),
                    nn.BatchNorm1d(num_features=256),
                    nn.ELU(),
                    nn.Dropout(p=0.5),
                    
                    # output
                    nn.Linear(in_features=256, out_features=num_targets)
        
        ) for num_targets in stn_targets])
    
    def forward(self, x):
        return tuple(fc(stn_out) for fc, stn_out in zip(self.fcls, self.mtn(x)))

class GenrePredictor(nn.Module):
    def __init__(self, mtn, num_targets):
        super().__init__()
        
        # the mtn is pretrained
        self.mtn = mtn.eval()
        
        for p in self.mtn.parameters():
            p.requires_grad_(False)
        
        self.fc = nn.Sequential(
                    
                    nn.Linear(in_features=STN.out_size * len(mtn.stns), out_features=1024),
                    nn.ELU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(in_features=1024, out_features=num_targets)
        
        )
    
    def train(self, mode=True):
        
        self.training = mode
        
        for module in self.children():
            
            if module == self.mtn:
                module.eval()
            else:
                module.train(mode)
        
        return self
    
    def forward(self, x):
        
        return self.fc(torch.cat(self.mtn(x), dim=1))
