import matplotlib.pyplot as plt
from pyquaternion import Quaternion


class BaseRender:
    """
    BaseRender class
    """

    def __init__(
            self,
            figsize=(10, 10)):
        self.figsize = figsize
        self.fig, self.axes = None, None

    def reset_canvas(self, dx=1, dy=1, tight_layout=False):
        plt.close()
        plt.gca().set_axis_off()
        plt.axis('off')
        self.fig, self.axes = plt.subplots(dx, dy, figsize=self.figsize)
        if tight_layout:
            plt.tight_layout()

    def close_canvas(self):
        plt.close()

    def save_fig(self, filename):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        print(f'saving to {filename}')
        plt.savefig(filename)
