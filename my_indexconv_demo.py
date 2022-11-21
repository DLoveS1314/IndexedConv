import torch

from indexedconv.engine import IndexedConv, IndexedAveragePool2d, IndexedMaxPool2d
from indexedconv.utils import neighbours_extraction ,prepare_mask

def test_conv():
    #N C L
    input = torch.randn(20, 16, 50)
    #  K L 
    indices = (10 * torch.rand(9, 50)).type(torch.LongTensor)
    # print(indices)  
    m = IndexedConv(16, 10, indices)
    output = m(input)
    print(f'input:{input.shape}')
    print(f'indices:{indices.shape}')
    
    # conv1= IndexedConv(1, 1, neighbours_indices)
    # out=conv1(data_1)
    print(f'output:{output.shape}')
def test_conv1():
    data_1 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7 ,8,9,10,11], dtype=torch.float).unsqueeze(0).unsqueeze(0)
    print('data_1',data_1.shape)
    
    # index_matrix = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]).unsqueeze(0).unsqueeze(0)
    index_matrix = torch.tensor([[0, 1, 2,3], [ 4, 5,6,7], [  8,9,10,11,]]).unsqueeze(0).unsqueeze(0)
    
    neighbours_indices = neighbours_extraction(index_matrix, 'Square',radius=1)
    print('neighbours_indices',neighbours_indices,neighbours_indices.shape)
    conv11 = IndexedConv(1, 1, neighbours_indices)
    out = conv11(data_1)
    print('out',out.shape)
    
test_conv1()

def test_mask():
    low  = 0
    high = 10
    size =[9]
    indices=torch.randint(low=low, high=high, size=size) 
    # indices = (torch.randint()).type(torch.LongTensor)
    print('indices',indices)
    new_indices, mask=prepare_mask(indices)
    print('new_indices',new_indices)    
    
    indices[8]=-1
    print('indices_0',indices)       
    new_indices, mask=prepare_mask(indices)
    print('new_indices_0',new_indices)
    # print()
# test_mask()
    
def test_unflod():
    import torch.nn as nn
    import torch
    kernel_size=[3,3]
    stride=2
    padding=0
    sampler = nn.Unfold(
        kernel_size=kernel_size,
        padding=padding,
        stride=stride)
    x=nn.rand()