import torch
import torch.nn as nn
import torch.nn.functional as F

from features import get_data_loaders, genre_counts, FramedFeatureDataset
from model import MTN, MTNFC
from logger import ModelLogger

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs, soft_targets):
        log_likelihood = F.log_softmax(inputs, dim=1)
        cross_entropy = -torch.sum(soft_targets * log_likelihood, dim=1)
        
        return torch.mean(cross_entropy)  # mean across batch

class MTNFCTrainer:
    
    def __init__(self, model, optimizer, batch_size,
                 weights, target_names, criterions,
                 alpha, logger, epochs, validate_every):
        
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        
        self.weights = weights
        self.target_names = target_names
        self.criterions = criterions
        
        self.alpha = alpha
        
        self.epochs = epochs
        self.validate_every = validate_every
        self.logger = logger
        
        self.dataset = FramedFeatureDataset()
        self.train_loader, self.valid_loader = get_data_loaders(self.dataset, self.batch_size, 0.15)
        
        self.dpmodel = nn.DataParallel(model)
        self.dpmodel.train()
        self.initial_losses = 0.0
        
        # log(C) init
        # self.initial_losses = torch.log(torch.tensor(self.model.stn_targets, dtype=torch.float, device='cuda'))
        
        self.best_loss = float('inf')
    
    def train(self):
        for epoch in range(self.epochs):
            
            print('Starting epoch', epoch + 1)
            
            # input is mel frames, labels is AGF targets
            for t, (input, labels) in enumerate(self.train_loader):
                
                self.step(epoch, t, input, labels)
                
                if t % (self.validate_every // self.batch_size) == 0:
                    self.report(epoch, t)
            
            print('Finished epoch {}/{}'.format(epoch + 1, self.epochs))
    
    def report(self, epoch, t):
        
        print('Epoch {}/{}, iteration {}/{}:'.format(epoch + 1, self.epochs, t + 1, len(self.train_loader)))
        
        for k, v in zip(self.logger.train_header, self.logger.train_buffer[-1]):
            print('{}: {}'.format(k, v))
        
        total_weighted_loss = self.validate()
        logger.log_valid(epoch, t, total_weighted_loss)
        
        print('Validation:')
        print('\tTotal weighted loss: {}'.format(total_weighted_loss))
        print()
        
        self.logger.flush()
        
        if total_weighted_loss < self.best_loss:
            print('Total weighted loss is better than previously best seen of {}'.format(self.best_loss))
            self.best_loss = total_weighted_loss
            self.logger.save(model, epoch, t, total_weighted_loss)
    
    def validate(self):
        
        with torch.no_grad():
            
            was_training = self.dpmodel.training
            
            if was_training:
                self.dpmodel.eval()
            
            all_losses = []  # total loss for every batch
            
            for i, (input, labels) in enumerate(self.valid_loader):
                input = input.cuda()
                targets = [labels[t].cuda() for t in self.target_names]
                
                # pass mel spectrogram as input
                task_outs = self.dpmodel(input)
                
                # compute task losses
                task_losses = tuple(crit(out, tar) for out, tar, crit in zip(task_outs, targets, self.criterions))
                task_losses = torch.stack(task_losses)
                
                # get the sum of weighted losses
                weighted_losses = self.weights * task_losses
                total_weighted_loss = weighted_losses.sum()
                
                all_losses.append(total_weighted_loss)
            
            all_losses = torch.stack(all_losses)
            
            if was_training:
                self.dpmodel.train()
            
            return all_losses.mean().item()
    
    def step(self, epoch, t, input, labels):
        
        input = input.cuda()
        targets = [labels[t].cuda() for t in self.target_names]
        
        # pass mel spectrogram as input
        task_outs = self.dpmodel(input)
        
        # compute task losses
        task_losses = tuple(crit(out, tar) for out, tar, crit in zip(task_outs, targets, self.criterions))
        task_losses = torch.stack(task_losses)
        
        # get the sum of weighted losses
        weighted_losses = self.weights * task_losses
        total_weighted_loss = weighted_losses.sum()
        
        self.optimizer.zero_grad()
        
        # compute and retain gradients
        total_weighted_loss.backward(retain_graph=True)
        
        # GRADNORM - learn the weights for each tasks gradients
        
        # zero the w_i(t) gradients since we want to update the weights using gradnorm loss
        self.weights.grad = 0.0 * self.weights.grad
        
        W = list(self.model.mtn.shared_block.parameters())
        norms = []
        
        for w_i, L_i in zip(self.weights, task_losses):
            # gradient of L_i(t) w.r.t. W
            gLgW = torch.autograd.grad(L_i, W, retain_graph=True)
            
            # G^{(i)}_W(t)
            norms.append(torch.norm(w_i * gLgW[0]))
        
        norms = torch.stack(norms)
        
        # set L(0)
        # if using log(C) init, remove these two lines
        if t == 0:
            self.initial_losses = task_losses.detach()
        
        # compute the constant term without accumulating gradients
        # as it should stay constant during back-propagation
        with torch.no_grad():
            
            # loss ratios \curl{L}(t)
            loss_ratios = task_losses / self.initial_losses
            
            # inverse training rate r(t)
            inverse_train_rates = loss_ratios / loss_ratios.mean()
            
            constant_term = norms.mean() * (inverse_train_rates ** self.alpha)
        
        # write out the gradnorm loss L_grad and set the weight gradients
        grad_norm_loss = (norms - constant_term).abs().sum()
        self.weights.grad = torch.autograd.grad(grad_norm_loss, self.weights)[0]
        
        # apply gradient descent
        self.optimizer.step()
        
        # renormalize the gradient weights
        with torch.no_grad():
            
            normalize_coeff = len(self.weights) / self.weights.sum()
            self.weights.data = self.weights.data * normalize_coeff
        
        # GRADNORM END
        
        # LOG PROGRESS
        
        self.logger.log_train(epoch, t,
                              total_weighted_loss.item(),
                              grad_norm_loss.item(),
                              *self.weights.tolist(),
                              *task_losses.tolist(),
                              *loss_ratios.tolist())

if __name__ == '__main__':
    target_names = ['genre', 'subgenres', 'mfcc', 'chroma', 'spectral_contrast']
    genre_weights = (1 / torch.tensor(genre_counts, dtype=torch.float)).cuda()
    
    criterions = [nn.CrossEntropyLoss(weight=genre_weights),
                  SoftCrossEntropyLoss(),
                  SoftCrossEntropyLoss(),
                  SoftCrossEntropyLoss(),
                  SoftCrossEntropyLoss()]
    
    C = len(genre_counts)
    mtn = MTN(num_stns=5)
    model = MTNFC(mtn=mtn, stn_targets=[C, 40, 40, 40, 40]).cuda()
    
    # weights for GradNorm
    weights = nn.Parameter(torch.ones(5, requires_grad=True, device='cuda'))
    
    # set differnent learning rate for GradNorm weights, and no weight decay
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5},
                                  {'params': weights, 'lr': 1e-3}])
    
    logger = ModelLogger(['epoch', 'iteration', 'total_weighted_loss', 'grad_norm_loss',
                          *(name + '_weight' for name in target_names),
                          *(name + '_loss' for name in target_names),
                          *(name + '_loss_ratio' for name in target_names)],
    
                         ['epoch', 'iteration', 'total_weighted_loss'],
                         'mtn_model')
    
    trainer = MTNFCTrainer(model=model, optimizer=optimizer, batch_size=64,
                           weights=weights, target_names=target_names, criterions=criterions,
                           alpha=0.5, logger=logger, epochs=30, validate_every=25000)
    
    trainer.train()