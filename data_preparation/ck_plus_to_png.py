import numpy as np

X = np.load('npy_files/X.npy')
y = np.load('npy_files/y.npy')

import matplotlib.pyplot as plt

emotions = {0:'neutral', 1:'angry', 2:'contempt', 3:'disgust', 4:'fear', 5:'happy', 6:'sad', 7:'surprise'}

for i in range(0,10):
    plt.xlabel(emotions[y[i]])
    plt.imshow(X[i].reshape(96,96),cmap='gray')
    plt.show()
    
# # from os import mkdir
# # for emotion in emotions:
# #     mkdir(f'C:/Users/Maryem/Desktop/ck+/ck-images' + f'{emotion} ' + f'{emotions[emotion]}')

# from PIL import Image

# count = 0
# for i in range(0,X.shape[0]):
#     count_string = str(count).zfill(7)
#     fname = 'C:/Users/Maryem/Desktop/ck+/ck-images' + f'{y[i]} ' + f'{emotions[y[i]]}/' + f'{emotions[y[i]]}-{count_string}.png'
#     image_array = X[i].astype(np.uint8)
# # Convert the NumPy array to an image object
#     image = Image.fromarray(image_array.reshape((96,96)))
#     image.save(fname) 
#     count += 1