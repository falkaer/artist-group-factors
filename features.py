import glob
import os
import torch
import warnings

import torch.multiprocessing
import torch.nn.functional as F

# this is really important - without this the program fails with a 
# "too many files open" error, at least on UNIX systems
torch.multiprocessing.set_sharing_strategy('file_system')

warnings.filterwarnings("ignore")

import numpy as np

from multiprocessing import cpu_count
import utils

audio_dir = os.path.join(os.path.curdir, 'fma_medium')
target_dir = os.path.join(audio_dir, 'targets')

os.environ['AUDIO_DIR'] = audio_dir

tracks = utils.load('tracks.csv')
genres = utils.load('genres.csv')

target_paths = [*glob.iglob(os.path.join(target_dir, '*_targets.npz'), recursive=True)]
tids = list(map(lambda x: int(os.path.splitext(os.path.basename(x).replace('_targets', ''))[0]), target_paths))

tracks_subset = tracks['track'].loc[tids]
genres_subset = tracks_subset['genre_top']
artists_subset = tracks['artist'].loc[tids]

from torch.utils.data import Dataset, DataLoader, random_split

genre_counts = genres_subset.value_counts()
genre_counts = genre_counts[genre_counts > 0]

print(genre_counts)

coded_genres = {genre: k for k, genre in enumerate(genre_counts.index)}
coded_genres_reverse = {k: genre for genre, k in coded_genres.items()}

print(coded_genres)

# X frames with 50% overlap = 2X-1 frames
num_frames = 4
total_frames = 2 * num_frames - 1
frame_size = 1290 // num_frames

class FeatureDataset(Dataset):
    def __len__(self):
        return len(target_paths)
    
    def __getitem__(self, idx):
        path = target_paths[idx]
        tid = tids[idx]
        
        # argmax these
        names = ['subgenres', 'mfcc', 'chroma', 'spectral_contrast']
        features = {}
        
        with np.load(path) as data:
            for k in names:
                # features[k] = data[k].argmax()
                features[k] = F.softmax(torch.from_numpy(data[k]), dim=0)
            
            mel = data['mel']
        
        features['genre'] = coded_genres[tracks_subset['genre_top'][tid]]
        
        return mel, features

class FramedFeatureDataset(FeatureDataset):
    def __len__(self):
        return total_frames * super().__len__()
    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        
        song_idx, frame = divmod(idx, total_frames)
        mel, features = super().__getitem__(song_idx)
        
        shift, half_shift = divmod(frame, 2)
        i = shift * frame_size + half_shift * frame_size // 2
        
        # add channel dimension so its 1x128x(frame_size)
        mel_frame = np.expand_dims(mel[:, i:i + frame_size], axis=0)
        
        return mel_frame, features

def get_data_loaders(dataset, batch_size, valid_split):
    dataset_len = len(dataset)
    
    # split dataset
    valid_len = int(dataset_len * valid_split)
    train_len = dataset_len - valid_len
    
    train_dataset, valid_dataset = random_split(dataset, [train_len, valid_len])
    
    # disable if it fucks things up but if it doesnt its apparently rly good 
    pin_memory = True
    
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=cpu_count(),
                              # sampler=train_sampler,
                              shuffle=True,
                              pin_memory=pin_memory)
    
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              num_workers=cpu_count(),
                              shuffle=False,
                              pin_memory=pin_memory)
    
    return train_loader, valid_loader

if __name__ == '__main__':
    
    dataset = FramedFeatureDataset()
    print(len(dataset))
    
    train_loader, valid_loader = get_data_loaders(dataset, 64, 0.15)
    print(len(train_loader))
    
    for _ in range(2):
        for i, batch in enumerate(train_loader):
            if i % 30 == 0:
                print(i, '/', len(train_loader))
