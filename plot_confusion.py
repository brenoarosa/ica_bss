import matplotlib.pyplot as plt
import numpy as np
import itertools
from matplotlib.colors import Normalize

def plot_confusion_matrix(cm, output_name,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap, norm=Normalize(vmin=0., vmax=1.))
    plt.title(title)
    plt.axis('off')
    plt.colorbar()

    thresh = .5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:.2f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_name)
