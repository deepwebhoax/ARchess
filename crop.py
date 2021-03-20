from PIL import Image
from chessboard_detection import inference, loadImage
import matplotlib.pyplot as plt
import numpy as np
import os 

def cropChessboard(im, tiles, file='temp_crops', padding = 12):
    np.empty((64,n,n))
    for tile in tiles:
        xyxy = [min(tile[0:2]), min(tile[2:4]), max(tile[0:2]), max(tile[2:4])]
        xyxy[0] -= padding
        xyxy[1] -= padding
        xyxy[2] += padding
        xyxy[3] += padding
        xyxy = [max(0, int(c)) for c in xyxy]    # preventing negative and float coordinates 
        cropped = im[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
        cropped = Image.fromarray(cropped)
        cropped.save(f'tiles/{k}.jpg')
        print(k)
        k+=1

folder = 'dataset/'
files = os.listdir(folder)
# path = 'dataset/4.jpg'

# params
padding = 12

k=0
for file in files:
    im = loadImage(folder+file)
    tiles = inference(folder+file)

    # plt.imshow(im, cmap='Greys_r')
    # for tile in tiles:
    #     plt.scatter(tile[0], tile[2])
    #     plt.scatter(tile[1], tile[3])
    # plt.show()

    for tile in tiles:
        xyxy = [min(tile[0:2]), min(tile[2:4]), max(tile[0:2]), max(tile[2:4])]
        xyxy[0] -= padding
        xyxy[1] -= padding
        xyxy[2] += padding
        xyxy[3] += padding
        xyxy = [max(0, int(c)) for c in xyxy]    # preventing negative and float coordinates 
        cropped = im[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
        cropped = Image.fromarray(cropped)
        cropped.save(f'tiles/{k}.jpg')
        print(k)
        k+=1

    
    
    
    # im.crop((x1, y2, x2-x1, y2-y1)).show()
    # np.save(f'tiles/{i}.jpg', cropped)
