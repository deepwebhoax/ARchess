import os
from PIL import Image
import pandas as pd


padding = 12
endfolder = 'dataset_marked/'
dir = 'pieces/'
pieces_folders = [d+'/' for d in os.listdir(dir)]
for folder in pieces_folders:
    files = os.listdir(folder)

    # df = pd.DataFrame({'image':[],'class_name':[]})

    # k = 530
    # for file in files:
    #     im = Image.open(folder+file)
    #     name = f"{k}.jpg"
    #     im.save(name)
    #     df = df.append({'image':f"{k}", 'class_name': 'empty'}, ignore_index=True)
    #     k+=1

    # df.to_csv('df_empty.csv')
    # print(df)



