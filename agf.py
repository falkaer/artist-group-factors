#!/usr/bin/python3
from collections import defaultdict

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import glob
import os
import utils
from gensim.models.ldamulticore import LdaMulticore

k_clusters = 2048
num_topics = 40

print('Loading tracks...')
tracks = utils.load('tracks.csv')

audio_dir = os.path.join(os.path.curdir, 'fma_medium')
raw_dir = os.path.join(audio_dir, 'raw')
target_dir = os.path.join(audio_dir, 'targets')

raw_paths = [*glob.iglob(os.path.join(raw_dir, '*.npz'), recursive=True)]
tids = list(map(lambda x: int(os.path.splitext(os.path.basename(x).replace('_raw', ''))[0]), raw_paths))

# SHAPES
# mel_frames (59, 128, 43)
# genres (1,)
# zcr (1, 1291)
# chroma (12, 513)
# spectral_contrast (7, 513)
# mfcc (12, 1290)

def to_corpus(X):
    return [[(i, x) for i, x in enumerate(doc)] for doc in X]

class AGFConstructor:
    
    def __init__(self):
        
        self.all_vectors = []
        
        self.artist_vectors = defaultdict(list)
        self.clustering = MiniBatchKMeans(n_clusters=k_clusters, batch_size=10000)
        
        self.artist_BoWs = {}
        self.artist_factors = {}
    
    def add_vectors(self, artist_id, X):
        
        self.all_vectors.append(X)
        self.artist_vectors[artist_id].append(X)
    
    def collect_vectors(self):
        
        self.all_vectors = np.concatenate(tuple(self.all_vectors), axis=1)
        self.artist_vectors = {k: np.concatenate(tuple(v), axis=1) for k, v in self.artist_vectors.items()}
    
    def train_kmeans(self):
        
        self.clustering.fit(self.all_vectors.T)
        del self.all_vectors
    
    # creates a BoW vector for every artist
    def make_BoWs(self):
        
        for artist_id, X in self.artist_vectors.items():
            pred = self.clustering.predict(X.T)
            BoW = np.bincount(pred, minlength=k_clusters)
            
            self.artist_BoWs[artist_id] = BoW
        
        del self.artist_vectors
        del self.clustering
    
    def make_factors(self):
        
        cor = to_corpus(self.artist_BoWs.values())
        lda = LdaMulticore(cor, num_topics=num_topics)
        
        for artist_id, BoW in self.artist_BoWs.items():
            # format that gensim expects
            formatted_BoW = [(i, x) for i, x in enumerate(BoW)]
            self.artist_factors[artist_id] = lda.inference([formatted_BoW])[0][0]
        
        del self.artist_BoWs

def make_subgenre_agf(agf):
    # for every artist, get list of song subgenres
    for artist_id, subgenre_lists in agf.artist_vectors.items():
        
        BoW = np.zeros(163, dtype=np.uint32)
        
        for subgenres in subgenre_lists:
            for subgenre in subgenres:
                BoW[subgenre] += 1
        
        agf.artist_BoWs[artist_id] = BoW
    
    agf.make_factors()

targets = ['mfcc', 'chroma', 'spectral_contrast', 'subgenres']
carry_over = ['mel']

agfs = {t: AGFConstructor() for t in targets}

print('Loading feature vectors...')

for i, (tid, path) in enumerate(zip(tids, raw_paths)):
    
    features = np.load(path)
    
    for t in targets:
        agfs[t].add_vectors(tracks['artist']['id'][tid], features[t])
    
    if (i + 1) % 50 == 0:
        print('Loaded', i + 1, 'vectors...')

targets.pop()  # remove subgenres from targets to build it directly
subg = agfs.pop('subgenres')
make_subgenre_agf(subg)

print('Loaded feature vectors, converting to numpy arrays...')

for t in targets:
    agfs[t].collect_vectors()
    print('Collected', t, 'shape:', agfs[t].all_vectors.shape)

for t in targets:
    print('Training k-means for target:', t)
    agfs[t].train_kmeans()
    
    print('Creating BoW vectors for target:', t)
    agfs[t].make_BoWs()
    
    print('Creating artist group factors for target:', t)
    agfs[t].make_factors()

targets.append('subgenres')
agfs['subgenres'] = subg

print('Saving all AGFs...')

import shutil

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

os.mkdir(target_dir)

for tid, path in zip(tids, raw_paths):
    
    out = {t: agfs[t].artist_factors[tracks['artist']['id'][tid]] for t in targets}
    
    # add mel quick and dirty
    f = np.load(path)
    
    for feature in carry_over:
        out[feature] = f[feature]
    
    np.savez(os.path.join(target_dir, str(tid) + '_targets.npz'), **out)
