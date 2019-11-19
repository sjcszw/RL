import torch
import torch.nn as nn

class Instructor():
    """ Builds functions for model training, evaluation, saving, loading
     
    evaluate(self, loader): elvauate the loss from loader data 
    """

    def __init__(self,model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        
    def train(self, loader, epoch, run_loss):
        """Trains the model for one epoch.
        Prints the epcoh number for print information
        Returns the ave loss.
        """
        losses = 0.0
        self.model.train()
        for t, (inp, out) in enumerate(loader): 
            self.optimizer.zero_grad()
            out_pre = self.model(inp)
            #loss = self.criterion(out_pre, out)
            loss = self.criterion(out_pre[:,:2], out[:,:2])
            run_loss += loss.data
            losses += loss.data
            if (t+1) % 100 == 0:
                print('[epoch: %d, %5d] training loss: %.3f' %
                      (epoch + 1, t + 1, run_loss / 100))
                run_loss = 0.0
            
            loss.backward()
            self.optimizer.step()        
        return losses/(t+1), run_loss

    
    def evaluate(self, loader):
        """Tests the model, returns the average loss."""
        losses = 0.0

        self.model.eval()
        with torch.no_grad():
            for n, (inp, out) in enumerate(loader):
                out_pre = self.model(inp)
                #loss = self.criterion(out_pre, out)
                loss = self.criterion(out_pre[:,:2], out[:,:2])
                losses += loss.data
        return losses/(n+1)
    
    def save(self,state,dir):
        """ Saves the model in location 'dir'.
        example:
        state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        dir = './model_trained/model_name'
        """
        if not os.path.isdir('model_trained'): #save model in a file "model_trainded"
            os.mkdir('model_trained')
        torch.save(state, dir)
        print('--- Save last model state')

    def load(self,dir):
        """ Loads the model in location 'dir', including net parameter, optimer, epoch number.
        Returns the next epoch number
        """
        if not os.path.isdir('model_trained'): #find the file for model saving and loading
            os.mkdir('model_trained')
        checkpoint = torch.load(dir)
        self.model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print('--- Load last model state')
        print('start epoch:',start_epoch)
        return start_epoch