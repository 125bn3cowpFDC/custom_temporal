import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
#x_ = feature.clone().detach()
#x = x_.expand(100,4,2)
class GCNLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.batchnorm = torch.nn.BatchNorm2d(100)
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = self.batchnorm(input)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        #output = self.batchnorm(output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output #렐루해조라



class Temporal_conv(torch.nn.Module): 
  def __init__(self,in_feature,d_rate=0.4,out_channels = 1,num_class=8):  
    super(Temporal_conv, self).__init__()  
    self.in_feature = in_feature 
    self.out_channels = out_channels
    self.num_class = num_class
    self.drop_rate  = d_rate
    self.conv_1 = torch.nn.Conv1d(in_channels=self.in_feature, out_channels=self.out_channels, kernel_size=4, stride=1)
    self.conv_2 = torch.nn.Conv1d(in_channels=self.in_feature, out_channels=self.out_channels, kernel_size=4, stride=1) 
    self.conv_3 = torch.nn.Conv1d(in_channels=self.in_feature, out_channels=self.out_channels, kernel_size=3, stride=1)
    self.conv_4 = torch.nn.Conv1d(in_channels=self.in_feature, out_channels=self.out_channels, kernel_size=3, stride=1)
    self.conv_5 = torch.nn.Conv1d(in_channels=self.in_feature, out_channels=self.out_channels, kernel_size=2, stride=1)
    self.conv_6 = torch.nn.Conv1d(in_channels=self.in_feature, out_channels=self.out_channels, kernel_size=2, stride=1)


    self.fc1 = torch.nn.Linear(in_features=18, out_features=num_class, bias=True)
    torch.nn.init.xavier_uniform_(self.fc1.weight)  
    #self.fc2 = torch.nn.Linear(in_features=10, out_features=self.num_class, bias=True)
    #torch.nn.init.xavier_uniform_(self.fc2.weight)  
  def forward(self, x):
    out1 = F.dropout(x,training=self.training,p=self.drop_rate)
    out1 = F.relu(self.conv_1(x))
    out1 = F.avg_pool1d(out1,kernel_size=95,stride=1)
    
    out2 = F.dropout(x,training=self.training,p=self.drop_rate)
    out2 = F.relu(self.conv_2(x))
    out2 = F.avg_pool1d(out2,kernel_size=95,stride=1)

    out3 = F.dropout(x,training=self.training,p=self.drop_rate)
    out3 = F.relu(self.conv_3(x))
    out3 = F.avg_pool1d(out3,kernel_size=96,stride=1)

    out4 = F.dropout(x,training=self.training,p=self.drop_rate)
    out4 = F.relu(self.conv_4(x))
    out4 = F.avg_pool1d(out4,kernel_size=96,stride=1)

    out5 = F.dropout(x,training=self.training,p=self.drop_rate)
    out5 = F.relu(self.conv_5(x))
    out5 = F.avg_pool1d(out5,kernel_size=97,stride=1)

    out6 = F.dropout(x,training=self.training,p=self.drop_rate)
    out6 = F.relu(self.conv_6(x))
    out6 = F.avg_pool1d(out6,kernel_size=97,stride=1)

    
    out = torch.cat((out1, out2, out3, out4, out5, out6), dim=1) 
    out = out.view(-1,18).contiguous()
    out = self.fc1(out)
    #out = self.fc2(out)
    #out = torch.log_softmax(self.fc(out))
    return out
  
if __name__ == "__main__":
    G = nx.Graph(name='G')

    for i in range(4):
        G.add_node(i, name=i)

    edges = [(0,1),(1,3),(1,2)]
    G.add_edges_from(edges)

    G_self_loops = G.copy()

    self_loops = []
    for i in range(G.number_of_nodes()):
        self_loops.append((i,i))

    G_self_loops.add_edges_from(self_loops)

    #Check the edges of G_self_loops after adding the self loops
    #print('Edges of G with self-loops:\n', G_self_loops.edges)

    #Get the Adjacency Matrix (A) and Node Features Matrix (X) of added self-lopps graph
    A_hat = np.array(nx.attr_matrix(G_self_loops, node_attr='name')[0])
    #print('Adjacency Matrix of added self-loops G (A_hat):\n', A_hat)

    D = np.diag(np.sum(A_hat, axis=0)) #create Degree Matrix of A
    D_half_norm = fractional_matrix_power(D, -0.5) #calculate D to the power of -0.5
    norm_ad = D_half_norm.dot(A_hat).dot(D_half_norm)
    norm_ad = torch.from_numpy(norm_ad).float()
    print(norm_ad)

    qq = torch.randn(3,100,4,2)

    model1 = GCNLayer(2,32)
    model2 = GCNLayer(32,16)
    model3 = GCNLayer(16,8)
    o1 = F.leaky_relu(model1(qq,norm_ad))
    #print(o1)
    o2 = F.leaky_relu(model2(o1,norm_ad))

    o2 = F.leaky_relu(model3(o2,norm_ad))
    #print(o2)
    #o2.view(-1,12)
    print(o2)

