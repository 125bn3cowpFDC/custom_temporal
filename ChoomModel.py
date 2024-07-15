import torch.nn as nn
import torch.nn.functional as F
from Layers import GCNLayer, Temporal_conv
import torch

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power

class Mymodel(nn.Module):
    def __init__(self, nfeat = 2, nhid = 16,d_rate=0.2, nhid_o=32,final_out=10,node=18):
        super(Mymodel, self).__init__()
        self.node = node
        self.final_out = final_out
        self.gcn_layer1 = GCNLayer(nfeat, nhid) #2 16
        #self.gcn_layer2 = GCNLayer(nhid, nhid_o) #16 32
        self.gcn_layer3 = GCNLayer(nhid, final_out) #16 10
        self.temporal_layer = Temporal_conv(node*final_out)
        self.dropout = d_rate

    def forward(self, x, adj):
        out = F.relu(self.gcn_layer1(x, adj))
        #out = F.relu(self.gcn_layer2(out, adj))
        out = F.relu(self.gcn_layer3(out, adj))
        out = F.dropout(out,training=self.training,p=self.dropout)
        #뒤집어
        out = out.view(-1,100,self.node*self.final_out).contiguous()
        out = out.permute(0,2,1).contiguous()
        out = self.temporal_layer(out)
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

    qq = torch.randn(5,100,4,2)

    model = Mymodel()
    out = model(qq,norm_ad)
    print(out) #fc뽑은거까지나온다 8개(클래스개수)


