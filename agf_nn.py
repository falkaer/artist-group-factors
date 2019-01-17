# nearest neighbours using AGFs
from glob import glob

import numba

import librosa

from model import MTN, MTNFC

import pandas as pd
import numpy as np
import os
import torch
import lws
import subprocess as sp

def load_audio(path):
    command = [
        'ffmpeg',
        '-i', path,
        '-f', 'f32le',  # float32
        '-acodec', 'pcm_f32le',  # float32
        '-ac', '1',  # mono
        '-ar', '22050',  # 22050 sampling rate
        '-'  # send output to stdout
    ]
    try:
        proc = sp.run(command, stdout=sp.PIPE, bufsize=10 ** 7, stderr=sp.DEVNULL, check=True)
        return np.frombuffer(proc.stdout, dtype=np.float32)
    except sp.CalledProcessError as e:
        print('Error occurred with file', path)
        print(e)

def get_mel(path):
    x = load_audio(path)
    
    n_fft = 1024
    hop_sz = 512
    sr = 22050
    
    _lws = lws.lws(n_fft, hop_sz, mode='music')
    
    S = np.abs(_lws.stft(x)).astype(np.float32)
    mel = librosa.amplitude_to_db(
                librosa.feature.melspectrogram(
                            sr=sr, S=S.T, n_fft=n_fft, hop_length=hop_sz
                )).astype(np.float32)
    
    # TODO: split?
    return mel

def load_best_mtn(model_log_path):
    maxid = -float('inf')
    
    for fname in glob(os.path.join(model_log_path, 'model_*.pt')):
        
        id = int(os.path.basename(fname)[6:-3])
        
        if id > maxid:
            maxid = id
    
    path = os.path.join(model_log_path, 'model_{}.pt'.format(maxid))
    return torch.load(path, map_location='cpu')['state_dict']

@numba.guvectorize(["void(float64[:], float64[:], float64[:])"], "(n),(n)->()", target='parallel')
def fast_cosine_gufunc(u, v, result):
    m = u.shape[0]
    udotv = 0
    u_norm = 0
    v_norm = 0
    for i in range(m):
        if (np.isnan(u[i])) or (np.isnan(v[i])):
            continue
        
        udotv += u[i] * v[i]
        u_norm += u[i] * u[i]
        v_norm += v[i] * v[i]
    
    u_norm = np.sqrt(u_norm)
    v_norm = np.sqrt(v_norm)
    
    if (u_norm == 0) or (v_norm == 0):
        ratio = 1.0
    else:
        ratio = udotv / (u_norm * v_norm)
    result[:] = ratio

C = 16
mtn = MTN(num_stns=5)
model = MTNFC(mtn=mtn, stn_targets=[C, 40, 40, 40, 40])
model.load_state_dict(load_best_mtn('./logs/mtn_model_1'))

model = model.eval().cuda()

print('model loaded')

agf_names = ['subgenres', 'mfcc', 'chroma', 'spectral_contrast']

agf_dir = os.path.join(os.path.curdir, 'agfs')
agfs = {k: pd.read_csv(os.path.join(agf_dir, k + '.csv'), header=None, index_col=0) for k in agf_names}

import utils

tracks = utils.load('tracks.csv')

while True:
    
    filename = input('File name: \n')
    path = os.path.join(os.path.curdir, filename)
    
    with torch.no_grad():
        
        mel = get_mel(path)
        mel = torch.from_numpy(mel).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
        
        pred = model(mel)[1:]
    
    dists = {}
    
    for i, name in enumerate(agf_names):
        u = pred[i].cpu().numpy()
        M = agfs[name].values
        
        dists[name] = fast_cosine_gufunc(u, M)
    
    total_dist = sum(dists.values())
    
    # get top N indices
    top_indices = np.argsort(-total_dist)[:10]
    artist_ids = agfs['subgenres'].iloc[top_indices].index.tolist()
    
    print('Top matching artists are:')
    
    for artist_id, index in zip(artist_ids, top_indices):
        
        name = tracks['artist'][tracks['artist']['id'] == artist_id].iloc[0]['name']
        print(name, total_dist[index] / 4)
        
        for k, v in dists.items():
            print('\t', k + '=' + str(v[index]))
