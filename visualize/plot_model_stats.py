import matplotlib.pyplot as plt


def plot_loss(loss_data: list[float]):
    plt.figure()
    plt.plot(loss_data)

    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.show()


def plot_accuracy(accuracy_data: list[float]):
    plt.figure()
    plt.plot(accuracy_data)

    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.show()
