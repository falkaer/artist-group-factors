import os
import glob
import warnings
import numpy as np
import lws
import subprocess as sp

#TODO: messy file, clean up? lmao no

# set this to music directory
audio_dir = 'fma_medium'
os.environ['AUDIO_DIR'] = os.path.join(os.path.curdir, audio_dir)

# import their custom utils.py - state path to utils.py
import utils

print('Loading tracks.csv and genres.csv...')
tracks = utils.load('tracks.csv')
genres = utils.load('genres.csv')
genres.reset_index(level=0, inplace=True)

# this is the sauce
class Extractor:
    """
    Input: Path to .mp3 file (path)
    Output: Two npz files:
        1) targets: fname_targets.npz
            - melspectrogram
            - genres
            - subgenres
        2) raw: fname_raw.npz
            - all other features
    """
    
    def __init__(self, target_dir, features, targets):
        
        self.target_dir = target_dir
        self.n_fft = 1024
        self.hop_sz = 512
        self._lws = lws.lws(self.n_fft, self.hop_sz, mode='music')
        
        if not targets:
            self.targets = ['genres', 'mel']
        else:
            self.targets = targets
        
        available_features = [
            'mel',
            'subgenres',
            'zcr',
            'chroma',
            'spectral_centroid',
            'spectral_bandwidth',
            'spectral_contrast',
            'spectral_rolloff',
            'mfcc']
        
        if not features:
            self.features = available_features
        else:
            self.features = [p for p in features if p in available_features]
    
    def load_audio(self, path):
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
    
    def compute_features(self, path, sr=22050):
        
        import librosa
        
        features_dict = dict()
        warnings.filterwarnings('error', module='librosa')
        x = self.load_audio(path)
        
        S = np.abs(self._lws.stft(x)).astype(np.float32)
        mel = librosa.amplitude_to_db(
                    librosa.feature.melspectrogram(sr=sr, S=S.T, n_fft=self.n_fft, hop_length=self.hop_sz)).astype(
                    np.float32)
        tid = int(os.path.splitext(os.path.basename(path))[0])
        mel = mel[:, :1290]
        
        # TODO: reduce mel spectograms float size to float16?
        
        if 'subgenres' in self.features:
            
            genre_ids = tracks['track']['genres_all'].loc[tid]
            indices = genres['genre_id'].loc[genres['genre_id'].isin(genre_ids)].index
            f = np.array(indices, dtype=np.uint32)
            
            features_dict['subgenres'] = f
        
        if 'mel' in self.features:
            # # drop the last 2 samples out of 1292 to be divisible by 43
            # frame_size = 43
            # num_splits = mel.shape[1] // frame_size
            # 
            # # drop excess samples (2 in case of 43 samples in frame) to be divisble by 43
            # mel = mel[:, :frame_size * num_splits]
            # frame_shift = frame_size // 2  # shift by half a frame for 50% overlap
            # sp1 = np.hsplit(mel, num_splits)
            # sp2 = np.hsplit(mel[:, frame_shift:frame_shift + frame_size * (num_splits - 1)], num_splits - 1)
            # f = np.array(sp1 + sp2)
            # 
            # assert f.shape == (59, 128, 43)
            # features_dict['mel_frames'] = f
            
            f = mel
            assert f.shape == (128, 1290)
            features_dict['mel'] = f
        
        # TODO: dMFCC?
        
        if 'chroma' in self.features:
            f = librosa.feature.chroma_stft(S=S ** 2, n_chroma=12)
            
            assert f.shape == (12, 513)
            features_dict['chroma'] = f.astype(np.float32)
        
        if 'mfcc' in self.features:
            # expects log-power mel spectrogram (amplitude-to-db computes log-power)
            f = librosa.feature.mfcc(S=mel, n_mfcc=12)
            
            assert f.shape == (12, 1290)
            features_dict['mfcc'] = f
        
        if 'spectral_contrast' in self.features:
            f = librosa.feature.spectral_contrast(S=np.abs(S))
            
            assert f.shape == (7, 513)
            features_dict['spectral_contrast'] = f.astype(np.float32)
        
        if 'spectral_centroid' in self.features:
            f = librosa.feature.spectral_centroid(S=np.abs(S))
            features_dict['spectral_centroid'] = f.astype(np.float32)
        
        if 'spectral_bandwidth' in self.features:
            f = librosa.feature.spectral_bandwidth(S=np.abs(S))
            features_dict['spectral_bandwidth'] = f.astype(np.float32)
        
        if 'spectral_rolloff' in self.features:
            f = librosa.feature.spectral_rolloff(S=S)
            features_dict['spectral_rolloff'] = f.astype(np.float32)
        
        return features_dict
    
    def write_feature_files(self, path):
        
        features = self.compute_features(path)
        
        basepath, _ = os.path.splitext(path)
        basename = os.path.basename(basepath)
        
        # for k, v in features_dict.items():
        #   print(k, v.dtype)
        
        # save all features
        np.savez(os.path.join(self.target_dir, basename + '_raw.npz'), **features)

from multiprocessing import cpu_count, Process, Queue

# %% If no arguements are given to the extractor it will:
# 1) calculate all possible features
# 2) write (genres, subgenres, mel_frames) features to target file
# 3) write all remaning features to raw (aka remainder) file

# %% The extractor takes two optional arguements
# Features to calculate and features to target file
# The difference between the two sets is written to the raw (or remainder) file
features = [  # all of these features will be calculated
    'mel',
    'subgenres',
    'chroma',
    'spectral_contrast',
    'mfcc']

# %% following features will be written to target file
targets = ['mel']
# reminaing features will be written to remainder file

song_paths = [*glob.iglob(os.path.join(audio_dir, '*/*.mp3'), recursive=True)]
target_dir = os.path.join(audio_dir, 'raw')

import shutil

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

os.mkdir(target_dir)

def apply_all():
    queue = Queue()
    
    for i, filename in enumerate(song_paths):
        queue.put((i, filename))
    
    count = i + 1
    
    def taker():
        
        while not queue.empty():
            
            i, filename = queue.get()
            
            try:
                e.write_feature_files(filename)
            except BaseException as ex:
                
                if type(ex) is IndexError:
                    raise ex
                
                print('Error extracting features for', filename, type(ex), ex)
            
            if (i + 1) % 50 == 0:
                print('Applied procedure to', i + 1, '/', count, 'files')
    
    ps = []
    
    for _ in range(cpu_count()):
        p = Process(target=taker)
        p.start()
        
        ps.append(p)
    
    print('Starting...')
    
    import signal
    
    def handler(sig, frame):
        
        for p in ps:
            p.terminate()
    
    signal.signal(signal.SIGINT, handler)
    
    for p in ps:
        p.join()

if __name__ == '__main__':
    
    multicore = True
    e = Extractor(target_dir, features, targets)
    
    if multicore:
        
        apply_all()
    
    else:
        
        for i, filename in enumerate(song_paths):
            
            try:
                print('Applying procedure to', filename)
                e.write_feature_files(filename)
                print('Applied procedure to', filename)
            except Exception as ex:
                print('Error extracting features for', filename, ex)
            
            if (i + 1) % 50 == 0:
                print('Applied procedure to', i + 1, 'files')
