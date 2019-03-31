from model.readFile import ReadData
from model.model import Model
from model.train import fit
from model.showcurve import ShowCurve
import torch
from torch.autograd import Variable

# hyper parameters
epochs = 1500000
show_epochs = 100000
learning_rate = 0.015
test_mode = 1   # calculate and show test loss
opt = "Adam" # set type of optimizer (with GD, SGD, ASGD, Adam )

# __main__
if epochs < show_epochs:
    print("ERROR!!:epochs should be larger than show_epochs")
else:
    x_train, y_train = ReadData("train")
    x_test, y_test = 0, 0
    if test_mode:
        x_test,y_test = ReadData("test")

    device = torch.device('cpu')
    model = Model().to(device)
    model.weight_bias_reset()   # initial weight and bias

    train_losses, test_losses = fit(x_train,y_train,model,opt=opt,learning_rate=learning_rate,epochs=epochs,show_epochs=show_epochs,device=device,test_mode=test_mode,x_test=x_test,y_test=y_test)
    # Show Curve
    ShowCurve(train_losses, "train_losses", 1)
    ShowCurve(test_losses, "test_losses", 1)
