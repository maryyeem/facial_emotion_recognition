import pandas as pd 
import numpy as np 
import csv
import os 

from PIL import Image

df=pd.read_csv('C:/Users/Maryem/Desktop/fer2013/fer2013_eval.csv')

emotions = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}

df['emotion'] =  [emotions[x] for x in df['emotion'] if x in emotions]

folder_path='C:/Users/Maryem/Desktop/fer2013/validation/'
# os.makedirs(folder_path,exist_ok=True)

# for emotion, label in emotions.items():
#     emotion_path=os.path.join(folder_path,f'{label}')
#     os.makedirs(emotion_path,exist_ok=True)

count = 0
for emotion,image_pixels in zip(df['emotion'], df['pixels']):
    image_string = image_pixels.split(' ') #pixels are separated by spaces
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    img = Image.fromarray(image_data) #final image\
    count_string = str(count).zfill(6) 

    img.save(folder_path + f'{emotion}/' + f'{emotion}-{count_string}.png') 
    count += 1
    
