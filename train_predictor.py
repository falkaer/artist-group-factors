import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from features import get_data_loaders, genre_counts, FramedFeatureDataset
from model import MTN, MTNFC, GenrePredictor
from logger import ModelLogger

from glob import glob

batch_size = 64
valid_split = 0.15

class PredictorTrainer:
    
    def __init__(self, model, optimizer, criterion, target_name, logger, epochs, validate_every):
        
        self.model = model
        self.optimizer = optimizer
        
        self.criterion = criterion
        self.target_name = target_name
        
        self.epochs = epochs
        self.validate_every = validate_every
        self.logger = logger
        
        self.dataset = FramedFeatureDataset()
        self.train_loader, self.valid_loader = get_data_loaders(self.dataset, batch_size, valid_split)
        
        self.dpmodel = nn.DataParallel(model)
        self.dpmodel.train()
        
        self.best_loss = float('inf')
        self.best_micro_f1 = 0.0
        self.best_mean_f1 = 0.0
    
    def train(self):
        for epoch in range(self.epochs):
            
            print('Starting epoch', epoch + 1)
            
            # input is mel frames, labels is AGF targets
            for t, (input, labels) in enumerate(self.train_loader):
                
                self.step(epoch, t, input, labels)
                
                if t % (self.validate_every // batch_size) == 0:
                    self.report(epoch, t)
            
            print('Finished epoch {}/{}'.format(epoch + 1, self.epochs))
    
    def report(self, epoch, t):
        
        print('Epoch {}/{}, iteration {}/{}:'.format(epoch + 1, self.epochs, t + 1, len(self.train_loader)))
        
        for k, v in zip(self.logger.train_header, self.logger.train_buffer[-1]):
            print('{}: {}'.format(k, v))
        
        loss, micro_f1, mean_f1 = self.validate()
        logger.log_valid(epoch, t, loss, micro_f1, mean_f1)
        
        print('Validation:')
        print('\tLog loss: {}'.format(loss))
        print('\tF1 score (micro): {}%'.format(micro_f1 * 100))
        print('\tF1 score (macro): {}%'.format(mean_f1 * 100))
        print()
        
        self.logger.flush()
        
        if loss < self.best_loss:
            print('Log loss is better than previously best seen of {}'.format(self.best_loss))
            self.best_loss = loss
            self.logger.save(model, epoch, t, loss, micro_f1, mean_f1)
        
        if micro_f1 > self.best_micro_f1:
            print('F1-score (micro) is better than previously best seen of {}%'.format(self.best_micro_f1 * 100))
            self.best_micro_f1 = micro_f1
            self.logger.save(model, epoch, t, loss, micro_f1, mean_f1)
        
        if mean_f1 > self.best_mean_f1:
            print('F1-score (mean) is better than previously best seen of {}%'.format(self.best_mean_f1 * 100))
            self.best_mean_f1 = mean_f1
            self.logger.save(model, epoch, t, loss, micro_f1, mean_f1)
    
    def validate(self):
        
        with torch.no_grad():
            
            was_training = self.dpmodel.training
            
            if was_training:
                self.dpmodel.eval()
            
            all_pred = []
            all_true = []
            all_losses = []  # loss for every batch
            
            for i, (input, labels) in enumerate(self.valid_loader):
                input = input.cuda()
                target = labels[self.target_name].cuda()
                
                # pass mel spectrogram as input
                out = self.dpmodel(input)
                loss = F.cross_entropy(out, target)  # cross entropy / log loss as validation measure
                
                all_pred.append(out.argmax(dim=1))
                all_true.append(target)
                all_losses.append(loss)
            
            all_pred = torch.cat(all_pred)
            all_true = torch.cat(all_true)
            all_losses = torch.stack(all_losses)
            
            if was_training:
                self.dpmodel.train()
            
            micro_f1 = f1_score(all_true, all_pred, average='micro')
            mean_f1 = f1_score(all_true, all_pred, average='macro')
            
            return all_losses.mean().item(), micro_f1, mean_f1
    
    def step(self, epoch, t, input, labels):
        
        input = input.cuda()
        target = labels[self.target_name].cuda()
        
        # pass mel spectrogram as input
        out = self.dpmodel(input)
        loss = self.criterion(out, target)
        
        self.optimizer.zero_grad()
        
        loss.backward()
        self.optimizer.step()
        
        # LOG PROGRESS
        self.logger.log_train(epoch, t, loss.item())

def load_best_mtn(model_log_path):
    maxid = -float('inf')
    
    for fname in glob(os.path.join(model_log_path, 'model_*.pt')):
        
        id = int(os.path.basename(fname)[6:-3])
        
        if id > maxid:
            maxid = id
    
    path = os.path.join(model_log_path, 'model_{}.pt'.format(maxid))
    return torch.load(path, map_location='cpu')['state_dict']

if __name__ == '__main__':
    genre_weights = (1 / torch.tensor(genre_counts, dtype=torch.float)).cuda()
    
    criterion = nn.CrossEntropyLoss(weight=genre_weights)
    
    C = len(genre_counts)
    mtn = MTN(num_stns=5)
    model = MTNFC(mtn=mtn, stn_targets=[C, 40, 40, 40, 40]).cuda()
    
    model.load_state_dict(load_best_mtn('./logs/mtn_model_1'))
    
    # we only need the mtn
    model = GenrePredictor(model.mtn, C).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    logger = ModelLogger(['epoch', 'iteration', 'log_loss'],
                         ['epoch', 'iteration', 'log_loss', 'micro_f1', 'mean_f1'],
                         'pred_model')
    
    trainer = PredictorTrainer(model=model, optimizer=optimizer, criterion=criterion, target_name='genre',
                               logger=logger, epochs=30, validate_every=25000)
    trainer.train()
