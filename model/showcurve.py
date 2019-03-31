import matplotlib.pyplot as plt
import numpy as np

def ShowCurve(ys, title, x_axis = 1, color = 'b'):
    if(x_axis == 0):
        return
    x_list = []
    for i in range(len(ys)):
        x_list.append(i * x_axis + 1)
    x = np.array(x_list)
    y = np.array(ys)
    plt.plot(x,y,c=color)

    plt.axis()
    plt.title('{} Curve:'.format(title))
    plt.xlabel('Epoch')
    plt.ylabel('{} Value'.format(title))
    plt.show()
