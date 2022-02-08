import matplotlib.pyplot as plt


def plot_loss(loss_data: list[float], show: bool = True, save_path: str = None):
    plt.figure()
    plt.plot(loss_data)

    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()


def plot_accuracy(accuracy_data: list[float], show: bool = True, save_path: bool = False):
    plt.figure()
    plt.plot(accuracy_data)

    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
