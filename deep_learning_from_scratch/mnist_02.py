import sys, os

sys.path.append(os.pardir)
from datasets.mnist import load_mnist
import numpy as np
from PIL import Image


def show_image(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)

first = x_train[0]
label = y_train[0]
first = first.reshape(28,-1)

print(label)
show_image(first)
