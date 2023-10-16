import numpy as np
import matplotlib.pyplot as plt

def view_classification(image, ps):
    """
    
    """
    ps = ps.data.numpy().squeeze()

    fig, [ax1, ax2] = plt.subplots(figsize=(16, 7), ncols=2)

    ax1.imshow(image.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off') 

    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_xlim(0, 1.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(['T-shirt/top',
                         'Trouser',
                         'Pullover',
                         'Dress',
                         'Coat',
                         'Sandal',
                         'Shirt',
                         'Sneaker',
                         'Bag',
                         'Ankle Boot'], size='small')
    ax2.set_title("Class Probability")

    plt.tight_layout()
    plt.show()

    return None