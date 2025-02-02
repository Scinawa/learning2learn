import torch.nn as nn





class SimpleNN(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, 3)
        self.layer2 = nn.Linear(3, 5)
        self.layer3 = nn.Linear(5, 2)
        self.layer4 = nn.Linear(2, 1)
        self.activation = activation()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.activation(self.layer4(x))
        return x




class BasicArchitecture(nn.Module):
    def __init__(self, 
                 nodes,
                 nn_architecture
                 nonlinearity="ReLU"       
                 ):
        super(SimpleNN, self).__init__()
        layers = []
        input_size = nn_architecture[0]

        nonlinearity = getattr(nn, nonlinearity)

        for i, output_size in enumerate(nodes):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nonlinearity())
            input_size = output_size
        layers.append(nn.Linear(input_size, nn_architecture[1]))
        self.fc = nn.Sequential(*layers)
        self.nonlin = nonlinearity
        self.fc2 = nn.Linear(nn_architecture[1], nn_architecture[2])
        self.out = nn.Linear(nn_architecture[2], nn_architecture[3])


    def forward(self, x):
        x = self.nonlin(self.fc(x))
        x = self.nonlin(self.fc2(x))
        x = self.nonlin(self.out(x))
        return x
