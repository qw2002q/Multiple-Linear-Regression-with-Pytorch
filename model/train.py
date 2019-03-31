import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def normalize(tensor):
    max1 = np.max(tensor.data[:,[0]].numpy())
    max2 = np.max(tensor.data[:,[1]].numpy())
    norl = torch.tensor([1/max1,1/max2])
    tensor = tensor*norl
    return tensor

def fit(x_data,y_data,model,opt="SGD",learning_rate=0.001,epochs=10000,show_epochs = 0,device=torch.device('cpu'),test_mode=0,x_test=0,y_test=0):
    loss_fn = nn.MSELoss(reduction='mean').to(device) # Cross entropy
    train_losses = [] # save train loss
    test_losses = [] # save test loss

    if opt == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif opt == "ASGD":
        optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate)
    elif opt == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif opt == "GD":
        # init weight &bias
        w = Variable(torch.randn(1,2), requires_grad=True)
        b = Variable(torch.zeros(1), requires_grad=True)
    else:
        print("ERROR!!: optimizer not found, only SGD/BGD available")
        return

    x_data = normalize(x_data) # Normalize!!!!
    x_data=Variable(x_data,requires_grad=True).to(device)
    y_data=Variable(y_data,requires_grad=True).to(device)

    if test_mode:   # test mode on
        x_test = normalize(x_test)
        x_test=Variable(x_test,requires_grad=True).to(device)
        y_test=Variable(y_test,requires_grad=True).to(device)

    for epoch in range(epochs):
        if opt == "GD":
            train_loss = train_GD(x_data,y_data,w,b,learning_rate,loss_fn,device=device)
        else:
            train_loss = train(x_data,y_data,optimizer,model,loss_fn,device=device)
        train_losses.append(train_loss)
        if test_mode: # test mode on
            if opt == "GD":
                test_loss = test_GD(x_test,y_test,w,b,loss_fn,device)
            else:
                test_loss = test(x_test,y_test,model,loss_fn,device)
            test_losses.append(test_loss)
        if show_epochs != 0:
            if epoch % show_epochs == 0:
                if test_mode:
                    print("step:",epoch+1,"train_loss:",train_loss,"test_loss:",test_loss)
                else:
                    print("step:",epoch+1,"train_loss:",train_loss)


    if test_mode:
        print("step:",epoch+1,"train_loss:",train_loss,"test_loss:",test_loss)
    else:
        print("step:",epoch+1,"train_loss:",train_loss)

    return train_losses, test_losses

# train_loader
def train(x_data,y_data,optimizer,model,loss_fn,device):
    model.train() # change the model into train mode
    optimizer.zero_grad()    # clear gradients of all optimized torch.Tensors'
    y_pred = model(x_data).to(device)  # make predictions
    loss = loss_fn(y_pred, y_data).to(device)   # calculate loss
    loss.backward() # compute gradient of loss over parameters
    optimizer.step() # update parameters with gradient descent
    return loss.item()

# test_loader
def test(x_test,y_test,model,loss_fn,device):
    model.eval() # change the model into test mode
    y_pred = model(x_test).to(device)
    loss = loss_fn(y_pred, y_test).to(device)
    return loss.item()

# train_loader for gradient descent
def train_GD(x_data,y_data,w,b,learning_rate,loss_fn,device):
    y_pred = nn.functional.linear(x_data,w,b)  # make predictions
    loss = loss_fn(y_pred, y_data).to(device)   # calculate loss
    loss.backward() # compute gradient of loss over parameters
    w.data = w.data - learning_rate*w.grad.data
    b.data = b.data - learning_rate*b.grad.data
    w.grad.zero_()
    b.grad.zero_()
    return loss.item()

# test_loader for gradient descent
def test_GD(x_test,y_test,w,b,loss_fn,device):
    y_pred = nn.functional.linear(x_test,w,b)
    loss = loss_fn(y_pred, y_test).to(device)
    w.grad.zero_()
    b.grad.zero_()
    return loss.item()

def Linear_GD(x_train,w,b):
    return (x_train[:,[0]]*w[0]+b[0])+(x_train[:,[1]]*w[1]+b[1])
