import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        pass

    def PlotEnumerated(self, y, y_label='error'):
        x = range(len(y))
        self.Plot(x,y,y_label)

    def Plot(self, x, y, y_label='error'):
        if len(x) != len(y):
            print('WARNING: incompatible x and y values for plotting')
            return
        plt.plot(x, y)
        plt.ylabel(y_label)
        plt.show()
