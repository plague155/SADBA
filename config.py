import torch
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
AGGR_MEAN = 'mean'
WHETHER_STORE=False
TYPE_MNIST='mnist'
TYPE_FashionMNIST='fashion_mnist'
TYPE_CIFAR10='cifar_10'



