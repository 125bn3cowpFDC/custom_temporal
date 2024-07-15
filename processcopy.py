from torch.utils.data import Dataset, DataLoader
import numpy as np
import Mymydata
import torch.nn as nn
import torch.nn.functional as F
from ChoomModel import Mymodel
import torch
from torchsummary import summary
import torch.optim as optim
import networkx as nx
import numpy as np
#import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
import time
import matplotlib.pyplot as plt
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()

def get_admatrix(device):
    sk_graph = nx.Graph(name='mediapipe_ver')

    for i in range(18): #node add
        sk_graph.add_node(i) 

    edges = [(9, 7), (7, 5), (10, 8), (8, 6), (16, 14), (14, 12), (15, 13), (13, 11),
          (12, 6), (11, 5), (6, 17), (5, 17), (0, 17), (2, 0), (1, 0), (4, 2),
          (3, 1)] # edge undirected by mediapipe custom input
    sk_graph.add_edges_from(edges)

    self_loops = sk_graph.copy()
    loops = []
    for i in range(sk_graph.number_of_nodes()):
        loops.append((i,i))

    self_loops.add_edges_from(loops)
    #Get the Adjacency Matrix
    A_hat = np.array(nx.attr_matrix(self_loops)[0])
    #print('Adjacency Matrix\n', A_hat)
    
    
    D = np.diag(np.sum(A_hat, axis=0))  #create Degree Matrix of A
    D_half_norm = fractional_matrix_power(D, -0.5) #calculate D to the power of -0.5
    norm_ad = D_half_norm.dot(A_hat).dot(D_half_norm) #systematic matrix
    norm_ad = torch.from_numpy(norm_ad).float().to(device)
    #print(norm_ad.shape)
    return norm_ad
    

def get_data(train,device):
    path = 'C:/Users/godma/OneDrive/Desktop/project/handmade_choom5/data_hanchoom_final'
    if train:
        dataset = Mymydata.SoobDataset(datapath=path, mode='train')
        dataloader = DataLoader(dataset=dataset,
                        batch_size=16,
                        shuffle=True,
                        drop_last=True,
                        num_workers=2)
    else:
        dataset = Mymydata.SoobDataset(datapath=path, mode='val')
        dataloader = DataLoader(dataset=dataset,
                        batch_size=16,
                        shuffle=True,
                        drop_last=False,
                        num_workers=2)
        
    admatrix = get_admatrix(device)

    return admatrix, dataloader

def train(log_interval, model, device, train_loader,admatrix, optimizer,epochs,val_loader):
    since = time.time() 
    acc_history = []
    loss_history = []
    val_acc_history = []
    val_loss_history = []
    
    best_acc = 0.0
    running_loss = 0.0
    running_corrects = 0

    val_best_acc = 0.0
    val_corrects = 0
    val_loss = 0.0

    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(epochs): 
        print('Epoch {}/{}'.format(epoch, epochs-1))
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0

        val_corrects = 0 #init
        val_loss = 0.0 #init

        for batch_idx, batch in enumerate(train_loader):
            data, target = batch
            data, label = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data,admatrix)
            #print(output.shape)
            loss = criterion(output, label)
            #writer.add_scalar("Loss/train", loss, batch_idx)
            predicted = torch.argmax(output.data, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()*data.size(0)
            print("running loss: ", loss.item()*data.size(0))
            running_corrects += torch.sum(predicted == label.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
           best_acc = epoch_acc

        acc_history.append(epoch_acc.item())
        loss_history.append(epoch_loss)

        val_acc_history, val_loss_history, val_best_acc = \
            val(model,device,val_loader,admatrix,val_acc_history,val_loss_history,val_best_acc,val_corrects,val_loss)



    torch.save(model.state_dict(), f'./te{epochs}st10lr0.0009b16bcafterbe.pt')

    print('Best Acc: {:4f}'.format(best_acc))
    print('Best Val acc :  {:4f}'.format(val_best_acc))
    return acc_history, loss_history, val_acc_history, val_loss_history


    
def val(model, device, val_loader, admatrix, val_acc_history, val_loss_history, val_best_acc, val_corrects,val_loss):
      

    criterion = nn.CrossEntropyLoss()
    model.eval()

    for batch_idx, batch in enumerate(val_loader):
        data, target = batch
        data, label = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data,admatrix)
        #print(output.shape)
        loss = criterion(output, label)
        #writer.add_scalar("Loss/train", loss, batch_idx)
        predicted = torch.argmax(output.data, 1)

        val_loss += loss.item()*data.size(0)
        #print("running loss: ", loss.item()*data.size(0))
        val_corrects += torch.sum(predicted == label.data)

    epoch_loss = val_loss / len(val_loader.dataset)
    epoch_acc = val_corrects.double() / len(val_loader.dataset)
    print(' val---Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    if epoch_acc > val_best_acc:
        val_best_acc = epoch_acc

    val_acc_history.append(epoch_acc.item())
    val_loss_history.append(epoch_loss)

    #torch.save(model.state_dict(), f'./te{epochs}st10.pt')

    #print('Best Acc: {:4f}'.format(best_acc))
    return val_acc_history, val_loss_history, val_best_acc


def main():
    log_interval =10
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    '''
    if platform.system() == 'Windows':
        nThreads =0 #if you use windows
    '''

    # datasets 
    admatrix, train_loader = get_data(train = True,device=device)
    admatrix, val_loader = get_data(train = False,device=device)
    
    # model
    model = Mymodel().to(device)
    #model.train()
    #summary(model)

    #optimizer = optim.SGD(model.parameters(), weight_decay=0,lr=0.0002, momentum=0.9)
    optimizer = optim.Adam(model.parameters(),lr=0.0009)
    
    from torchsummary import summary
    import pytorch_model_summary
    #from torchinfo import summary
    ##summary(model,[(100,18,2),(100,18,18)])
    #print(pytorch_model_summary.summary(model,[(100,18,2),(18,18)],max_depth=1,show_parent_layers=True,show_input=True,show_hierarchical=True))
    #summary(model,[(100,18,2),(100,18,18)])
    #from torchviz import make_dot
    #x = torch.randn(1, 100,18,2,requires_grad=True).to(device)

    #y = model(x,admatrix)

    #make_dot(y.mean(), params=dict(model.named_parameters())).render("./model_archi",format='png')

    
    train_acc, train_loss, val_acc, val_loss = train(log_interval, model, device, train_loader, admatrix, optimizer, 50,val_loader)
    
    ephocs = []
    for i in range(50):
        ephocs.append(i+1)
    
    plt.figure(figsize=(10.0, 3.0))
    
    plt.subplot(1, 2, 1)
    plt.title("Accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel("Ephoc")
    plt.plot(ephocs,train_loss,label = "train_loss")
    plt.plot(ephocs,val_loss,label = "val_loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.ylabel('max')
    plt.xlabel("Ephoc")
    plt.plot(ephocs,train_acc,label = "train_acc")
    plt.plot(ephocs,val_acc,label = "val_acc")
    plt.legend()

    plt.tight_layout()
    plt.show()

    
    '''
    test_dataset = Mymydata.SoobDataset(datapath='C:/Users/godma/OneDrive/Desktop/project/handmade_choom5/data_hanchoom_final'
                          , mode="test")
    
    test_dataloader = DataLoader(dataset=test_dataset,
                        batch_size=1,
                        shuffle=False,
                        drop_last=False,
                        num_workers=2)
    
    datas,labels=(next(iter(test_dataloader)))
    datas, labels = datas.to(device), labels.to(device)
    model.eval()
    with torch.no_grad():
        print('라벨',labels)
        model.load_state_dict(torch.load('./te50st10lr0.0009b16_best.pt'))

        outputs = model(datas,admatrix)
        softmax_out = torch.nn.functional.softmax(outputs, dim=1)*100
        softmax_out = softmax_out.to('cpu').numpy()
        np.set_printoptions(precision=2, suppress=True)
        predicted = torch.argmax(outputs,1)
        print('예측',predicted[0].item())
        print(softmax_out)
    '''
if __name__ == "__main__":
    main()

    