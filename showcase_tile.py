import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

dir = 'tiles/'
files = os.listdir(dir)
f, ax = plt.subplots(8,8)
i = 0
for file in files:
    if i==64:
        break
    im = Image.open(dir+file)
    ax[i//8][i%8].imshow(cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB))
    i+=1


plt.show()