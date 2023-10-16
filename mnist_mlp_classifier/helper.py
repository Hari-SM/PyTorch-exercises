import matplotlib.pyplot as plt
import numpy as np

def view_classification(image, ps):
    """
    A function to view an image with the predicted class probabilities

    :param image: A flattened tensor with 784 entries
    :param ps: A 1D tensor of probabilities list (=10 elements)
    :return None: This will just produce an image  
    """
    ps = ps.data.numpy().squeeze()

    fig, [ax1, ax2] = plt.subplots(figsize=(16, 9), ncols=2)

    ax1.imshow(image.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title("Class Probability")
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()

    return None