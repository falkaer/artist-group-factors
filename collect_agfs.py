import glob
import shutil

import numpy as np
import pandas as pd
import os

import utils

audio_dir = os.path.join(os.path.curdir, 'fma_medium')
target_dir = os.path.join(audio_dir, 'targets')

target_paths = [*glob.iglob(os.path.join(target_dir, '*_targets.npz'), recursive=True)]
tids = list(map(lambda x: int(os.path.splitext(os.path.basename(x).replace('_targets', ''))[0]), target_paths))

tracks = utils.load('tracks.csv')
df = tracks['artist'].groupby('name').first()

agfs = {
    
    'chroma'           : {},
    'mfcc'             : {},
    'spectral_contrast': {},
    'subgenres'        : {}
    
}

for i, (path, tid) in enumerate(zip(target_paths, tids)):
    
    artist = tracks['artist'].loc[tid]
    artist_id = artist['id']
    
    if artist_id not in agfs['chroma']:
        with np.load(path) as data:
            for k, v in agfs.items():
                v[artist_id] = data[k]
    
    if i % 50 == 0:
        print('Checked file', i, '/', len(tids))
    
agf_path = os.path.join(os.path.curdir, 'agfs')

if os.path.exists(agf_path):
    shutil.rmtree(agf_path)

os.mkdir(agf_path)

for k, v in agfs.items():
    
    df = pd.DataFrame.from_dict(v, orient='index')
    df.to_csv(os.path.join(agf_path, k + '.csv'), header=False)
    
