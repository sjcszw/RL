import os

import torch

def save_checkpoint(state, filename):
    torch.save(state, filename)

def save_model(trainer,filename='../model/checkpoint.pth.tar'):
    save_checkpoint({
        'step_total': trainer.step_total,
        'state_dict': trainer.current_net.state_dict(),
        'optimizer' : trainer.optimizer.state_dict(),
        'trac_reward':trainer.trac_reward,
        'trac_time':trainer.trac_time,
        'losses':trainer.losses,
        'memory':trainer.memory
    }, filename)
    
def load_model(trainer,filename='../model/checkpoint.pth.tar'):
    if filename:
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            trainer.step_total = checkpoint['step_total']
            trainer.current_net.load_state_dict(checkpoint['state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer'])
            trainer.trac_reward = checkpoint['trac_reward']
            trainer.trac_time = checkpoint['trac_time']
            trainer.losses = checkpoint['losses']
            trainer.memory = checkpoint['memory']
    return trainer
