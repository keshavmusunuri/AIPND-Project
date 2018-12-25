import torch
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self,input_size,hidden_sizes,output_size,dropout_prob = 0.5):
        super().__init__()
        
        self.layers = nn.ModuleList([nn.Linear(input_size,hidden_sizes[0])])
        layer_sizes = zip(hidden_sizes[:-1],hidden_sizes[1:])
        self.layers.extend([nn.Linear(x,y) for x,y in layer_sizes])
        self.output = nn.Linear(hidden_sizes[-1],output_size)
        self.dropout = nn.Dropout(p = dropout_prob)
        
        
    def forward(self,x):
        
        x = x.view(x.shape[0],-1)
        for linear_transform in self.layers:
            x = self.dropout(F.relu(linear_transform(x)))
        x = self.output(x)
        
        return F.log_softmax(x,dim=1)
 