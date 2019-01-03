from multiprocessing import cpu_count

import torch
from torch.utils.data import DataLoader

torch.cuda.set_device(1)

from features import FeatureDataset, coded_genres
import pandas as pd

from features import genre_counts
from model import MTN, GenrePredictor

C = len(genre_counts)

# LOAD STATE DICT
model_path = './logs/pred_model_0/model_73.pt'
model_state = torch.load(model_path, map_location=torch.device('cpu'))

record = model_state['record']
state_dict = model_state['state_dict']

print(record)

class SlicedGenrePredictor(GenrePredictor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sliced_fc = list(self.fc.children())[:-1]
    
    def forward(self, x):
        # add channel dimension
        x = x.unsqueeze(1)
        x = torch.cat(self.mtn(x), dim=1)
        
        for module in self.sliced_fc:
            x = module(x)
        
        return x

if __name__ == '__main__':
    
    with torch.no_grad():
        
        reverse_onehot_genres = {v: k for k, v in coded_genres.items()}
        
        mtn = MTN(num_stns=5)
        model = SlicedGenrePredictor(mtn=mtn, num_targets=C)
        model.load_state_dict(state_dict)
        
        model = model.eval().cuda()
        
        dataset = FeatureDataset()
        loader = DataLoader(dataset,
                            batch_size=64,
                            num_workers=cpu_count(),
                            # sampler=train_sampler, 
                            shuffle=False,
                            pin_memory=True)
        
        import numpy as np
        
        slices_list = []
        labels_list = []
        
        for t, (input, labels) in enumerate(loader):
            
            input = input.cuda()
            
            slices = model(input)
            slices = slices.cpu().numpy()
            genres = labels['genre'].cpu().numpy()
            
            for s, l in zip(slices, genres):
                slices_list.append(s)
                labels_list.append(reverse_onehot_genres[l])
            
            if t % 25 == 0:
                print(t, '/', len(loader), '...')
        
        slice_df = pd.DataFrame(data=np.stack(slices_list), columns=[i for i in range(1024)])
        labels_df = pd.DataFrame(data=np.asarray(labels_list).reshape(-1, 1), columns=[0])
        
        slice_df.to_csv('./slices.csv', index=False)
        labels_df.to_csv('./labels.csv', index=False)
