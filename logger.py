import os
import torch
from itertools import count

def write_header(fname, header):
    
    with open(fname, 'w') as f:
        f.write(','.join(str(a) for a in header) + '\n')
        
def get_next_path(basepath, ext):
    for i in count():
        
        outpath = os.path.join(basepath + '_' + str(i) + (ext if ext is not None else ''))
        
        if not os.path.exists(outpath):
            return outpath

class ModelLogger:
    log_base_path = os.path.join(os.path.curdir, 'logs')
    
    def __init__(self, train_header, valid_header, name):
        
        self.log_path = get_next_path(os.path.join(self.log_base_path, name), None)
        os.makedirs(self.log_path)
        
        print('Logger based in', self.log_path)
        
        self.train_log_path = os.path.join(self.log_path, 'train_log.csv')
        self.valid_log_path = os.path.join(self.log_path, 'valid_log.csv')
        
        self.train_header = train_header
        self.valid_header = valid_header
        
        write_header(self.train_log_path, self.train_header)
        write_header(self.valid_log_path, self.valid_header)
        
        self.train_buffer = []
        self.valid_buffer = []
    
    def log_train(self, *record):
        self.train_buffer.append(record)
    
    def log_valid(self, *record):
        self.valid_buffer.append(record)
    
    def flush(self):
        
        with open(self.train_log_path, 'a') as f:
            
            for record in self.train_buffer:
                f.write(','.join(str(a) for a in record) + '\n')
        
        with open(self.valid_log_path, 'a') as f:
    
            for record in self.valid_buffer:
                f.write(','.join(str(a) for a in record) + '\n')
        
        self.train_buffer.clear()
        self.valid_buffer.clear()
    
    def save(self, model, *record):
        
        model_path = get_next_path(os.path.join(self.log_path, 'model'), '.pt')
        
        torch.save({
            
            'record': record,
            'state_dict': model.state_dict()
            
        }, model_path)
        
        return model_path
