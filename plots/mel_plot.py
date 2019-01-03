import subprocess as sp

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
import librosa.display
import seaborn as sns
import lws

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
        return np.fromstring(proc.stdout, dtype=np.float32)
    
    except sp.CalledProcessError as e:
        
        print('Error occurred with file', path)
        print(e)


x = load_audio('../fma_medium/018/018350.mp3')

start, end = 0, 30
sr = 22050

n_fft = 1024
hop_sz = 512

_lws = lws.lws(n_fft, hop_sz, mode='music')
S = np.abs(_lws.stft(x)).astype(np.float32)
mel = librosa.amplitude_to_db(librosa.feature.melspectrogram(sr=sr, S=S.T, n_fft=n_fft))

f, ax = plt.subplots(figsize=(10, 6), dpi=100)

# # do a frame instead
# print(mel.shape[1] // 7)
# sns.heatmap(mel[:,0:mel.shape[1] // 7], cmap=cm.rainbow, cbar=False)
# sns.despine(f, ax, right=True, left=True, top=True, bottom=True, trim=True)
# plt.xticks([])
# plt.yticks([])
# f.show()

print(mel.shape)
sns.heatmap(mel, cmap=cm.rainbow, cbar=False)
sns.despine(f, ax, right=True, left=True, top=True, bottom=True, trim=True)
plt.xticks([])
plt.yticks([])
f.show()