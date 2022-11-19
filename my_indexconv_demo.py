import torch

from indexedconv.engine import IndexedConv, IndexedAveragePool2d, IndexedMaxPool2d
from indexedconv.utils import neighbours_extraction ,prepare_mask

def test_conv():
    #N C L
    input = torch.randn(20, 16, 50)
    #L K    
    indices = (10 * torch.rand(50, 9)).type(torch.LongTensor)
    # print(indices)  
    m = IndexedConv(16, 9, indices)
    output = m(input)
    print(f'input:{input.shape}')
    print(f'indices:{indices.shape}')
    
    # conv1= IndexedConv(1, 1, neighbours_indices)
    # out=conv1(data_1)
    print(f'output:{output.shape}')
# test_conv()

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
test_mask()
    