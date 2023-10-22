import os
import time
import torch
import pickle as pk
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import MyModel
from dataloader import MyDataloader
from constant import CONSTANT


C = CONSTANT()


def train(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    top1_accu, top3_accu, cnt = 0, 0, 0
    for x,y in loader:
        optimizer.zero_grad()
        x,y = x.to(C.device),y.to(C.device)
        yhat = model(x)
        loss = loss_fn(yhat,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # Add your own metrics here
        # accu
        y_idx = torch.topk(yhat, k=3, dim=1).indices
        top1_accu += torch.sum(y_idx[:,0] == y).item()
        for i in range(3):
            top3_accu += torch.sum(y_idx[:,i] == y).item()
        cnt += len(y)
    return total_loss/len(loader), top1_accu/cnt, top3_accu/cnt


def test(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    top1_accu, top3_accu, cnt = 0, 0, 0
    for x,y in loader:
        x,y = x.to(C.device),y.to(C.device)
        yhat = model(x)
        loss = loss_fn(yhat,y)
        total_loss += loss.item()
        # Add your own metrics here
        # accu
        y_idx = torch.topk(yhat, k=3, dim=1).indices
        top1_accu += torch.sum(y_idx[:,0] == y).item()
        for i in range(3):
            top3_accu += torch.sum(y_idx[:,i] == y).item()
        cnt += len(y)
    return total_loss/len(loader), top1_accu/cnt, top3_accu/cnt


def myplot(config):
    plt.title(config['title'])
    plt.xlabel(config['xlabel'])
    plt.ylabel(config['ylabel'])
    for label in config['data']:
        plt.plot(config['data'][label][0], config['data'][label][1], label=label)
    plt.legend()
    plt.savefig(config['savefig'])
    plt.clf()


def test_model(datePath):
    # Model
    model = MyModel()
    model.load_state_dict(torch.load(datePath+'/model.pt'))
    model = model.to(C.device)
    model.eval()

    '''Test'''
    # Data
    dataloaders = MyDataloader()
    dataloaders.setup(['test'])
    loader = dataloaders.loader['test']
    artists = dataloaders.artists
    scoreDict = {}

    for x,y in loader:
        x,y = x.to(C.device),y.to(C.device)
        yhat = model(x)
        # map index and write result
        for row in range(y.shape[0]):
            label = y[row].detach().cpu().item()
            score = yhat[row].detach().cpu()
            scoreDict[int(label)] = score if int(label) not in scoreDict else scoreDict[int(label)] + score

    with open(datePath+'/result.csv','w') as f:         
        for key in sorted(scoreDict.keys()):
            yhat = scoreDict[key]
            y_idx = torch.topk(yhat, k=3).indices
            aPredict = [artists[y_idx[i]] for i in range(3)]
            line = f'{key},{aPredict[0]},{aPredict[1]},{aPredict[2]}\n'
            f.write(line)

    ''' Valid '''
    # # Data
    # dataloaders = MyDataloader()
    # dataloaders.setup(['valid'])
    # loader = dataloaders.loader['valid']
    # artists = dataloaders.artists

    # with open(datePath+'/valid_result.csv','w') as f:
    #     for x,y in loader:
    #         x,y = x.to(C.device),y.to(C.device)
    #         yhat = model(x)
    #         y_idx = torch.topk(yhat, k=3, dim=1).indices
    #         # map index and write result
    #         for row in range(y.shape[0]):
    #             aPredict = [artists[y_idx[row,i]] for i in range(3)]
    #             line = f'{artists[y[row].detach().item()]},{aPredict[0]},{aPredict[1]},{aPredict[2]}\n'
    #             f.write(line)
    return



def main():
    # Load model and data
    model = MyModel()
    model = model.to(C.device)
    dataloaders = MyDataloader()
    dataloaders.setup(['train', 'valid', 'test'])

    # You can adjust these as your need
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=C.milestones, gamma=C.gamma)

    # Set up output directory
    start_time = str(time.strftime("%Y-%m-%d~%H:%M:%S", time.localtime()))
    if not os.path.exists('output'):
        os.mkdir('output')
    os.mkdir('output/%s'%start_time)
    with open('output/%s/log.txt'%start_time, 'w') as f:
        for key, value in vars(C).items():
            f.write(f"{key}: {value}\n")
        f.write('Optimizer:'+str(optimizer)+'\n')
        f.write('Scheduler:'+str(scheduler)+'\n')
        f.write('\n===== Model Structure =====')
        f.write('\n'+str(model)+'\n')
        f.write('===== Begin Training... =====\n')

    # Start training
    train_losses = []
    valid_losses = []
    train_accus = []
    valid_accus = []
    train_accu3s = []
    valid_accu3s = []

    p_cnt = 0
    best_valid_loss = 1e10
    best_valid_accu = 0
    best_valid_accu3 = 0

    for e in tqdm(range(1,1+C.epochs)):
        train_loss, train_accu, train_accu3 = train(model, dataloaders.loader['train'], optimizer, loss_fn)
        valid_loss, valid_accu, valid_accu3 = test(model, dataloaders.loader['valid'], loss_fn)

        scheduler.step()
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accus.append(train_accu)
        valid_accus.append(valid_accu)
        train_accu3s.append(train_accu3)
        valid_accu3s.append(valid_accu3)
        print(f'Epoch = {e}, Train / Valid Loss = {round(train_loss, 6)} / {round(valid_loss, 6)}, ACCU = {round(train_accu, 4)} / {round(valid_accu, 4)}', end='')
        print(f', TOP 3 ACCU = {round(train_accu3, 4)} / {round(valid_accu3, 4)}')

        if e % C.verbose == 0:
            print('Plotting Loss at epoch', e)
            x_axis = list(range(e))
            config = {
                'title':'Loss',
                'xlabel':'Epochs',
                'ylabel':'Loss',
                'data':{
                    'Train':[x_axis, train_losses],
                    'Valid':[x_axis, valid_losses]
                },
                'savefig':'output/%s/loss.png'%start_time
            }
            myplot(config)
            
            config = {
                'title':'Accuracy',
                'xlabel':'Epochs',
                'ylabel':'Accuracy',
                'data':{
                    'Train':[x_axis, train_accus],
                    'Valid':[x_axis, valid_accus]
                },
                'savefig':'output/%s/accu.png'%start_time
            }
            myplot(config)

            config = {
                'title':'Accuracy (top 3)',
                'xlabel':'Epochs',
                'ylabel':'Accuracy',
                'data':{
                    'Train':[x_axis, train_accu3s],
                    'Valid':[x_axis, valid_accu3s]
                },
                'savefig':'output/%s/accu3.png'%start_time
            }
            myplot(config)

        # Save best model and early stopping
        if valid_loss < best_valid_loss:
            p_cnt = 0
            best_valid_loss = valid_loss
            best_valid_accu = valid_accu
            best_valid_accu3 = valid_accu3
            torch.save(model.state_dict(), 'output/%s/model.pt'%start_time)
        else:
            p_cnt += 1
            if p_cnt == C.patience:
                print('Early Stopping at epoch', e)
                break

        with open('output/%s/losses.pickle'%start_time, 'wb') as file:
            d = {'loss': [train_losses, valid_losses, best_valid_loss], 'accu' : [train_losses, valid_losses, best_valid_loss]}
            pk.dump(d, file)
        
        with open('output/%s/log.txt'%start_time, 'a') as f:
            f.write(f"Epoch {e}: Best valid loss: {round(best_valid_loss, 6)}, Best valid accuracy: {round(best_valid_accu, 4)} / {round(best_valid_accu3, 4)}\n")

    message = f'Ending at epoch {e}. Best valid loss: {round(best_valid_loss, 6)}, Best valid accuracy: {round(best_valid_accu, 4)} / {round(best_valid_accu3, 4)}'
    print(message)
    with open('output/%s/log.txt'%start_time, 'a') as f:
        f.write(message)


if __name__ == '__main__':
    # main()
    test_model('output/2023-10-10~21:13:49')

