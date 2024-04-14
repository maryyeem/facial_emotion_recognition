from helpers import * 
import io 
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
import sys
from tensorflow.keras.models import load_model 
from sklearn.preprocessing import MinMaxScaler

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util

scope = 'user-library-read'
client_id='0ba987987f94457694febf1b01b1d156'
client_secret='f990187a67e746419d728f9b43dd994a'
redirect_uri='https://open.spotify.com/'
username = 'your_username_spotify_code'


# auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
# sp = spotipy.Spotify(auth_manager=auth_manager)
# token = util.prompt_for_user_token(scope, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)
# sp = spotipy.Spotify(auth=token)
# if len(sys.argv) > 1:
#     username = sys.argv[1]
# else:
#     print("Usage: %s username" % (sys.argv[0],))
#     sys.exit()

#gather playlist names and images. 
#images aren't going to be used until I start building a UI


sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id, client_secret=client_secret,scope=scope,redirect_uri=redirect_uri))
token = util.prompt_for_user_token(scope, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)
target={0:"calm",1:"energetic",2:"happy",3:"sad"}
results = sp.current_user_saved_tracks(limit=5)
id_name={}
list_photo={}
for idx, item in enumerate(results['items']):
    track = item['track']
    id_name[track['name']] = track['uri'].split(':')[2]
    list_photo[track['uri'].split(':')[2]] = track['album']['images'][0]['url']
    # print(idx, track['id])


def predict_mood(id_song):
    # #Join the model and the scaler in a Pipeline
    # pip = Pipeline([('minmaxscaler',MinMaxScaler()),('keras',KerasClassifier(build_fn=baseModel,epochs=300,batch_size=200,verbose=0))])
    # #Fit the Pipeline
    # pip.fit(X1,dummy)
    music_model=load_model('music_model.h5')
    #Obtain the features of the song
    preds = get_songs_features(id_song)
    #Pre-process the features to input the Model
    preds_features = np.array(preds[0][6:-2]).reshape(-1,1).T
    preds_features = MinMaxScaler().fit_transform(preds_features)
    #Predict the features of the song
    results = music_model.predict(preds_features)
    mood_index=np.argmax(results)
    mood = target[mood_index]
    name_song = preds[0][0]
    artist = preds[0][2]

    return print("{0} by {1} is a {2} song".format(name_song,artist,mood.upper()))
# from urllib.request import urlopen
# from PIL import Image

for id_track in id_name.values():
    predict_mood(id_track)
    print('\n')
# import IPython
# for url in list_photo.values():
#     img = Image.open(urlopen(url))
#     img.show()
