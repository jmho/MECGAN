import streamlit as st

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import io
import urllib, base64
from math import sqrt
import onnxruntime

# GLOBALS 
num_classes = 8
latent_dim = 128

st.title('MECGAN Test Environment')

st.write('Welcome! This is a test environment to explore the GAN we developed for CIS4914. The GAN is supposed to take in a label from an ANN that classifies music based on a user\'s Spotify choice but for your convenience just the GAN is attatched. Select a mood to see the magic happen.')

mood = st.selectbox('Select a mood you want to generate!',['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness'])

def generate_latent_points(latent_dim, n_class):
    labels = np.eye(num_classes)[n_class]
    noise = np.random.normal(size=(1, latent_dim))
    return np.concatenate((noise, [labels]), 1).astype(np.float32)

def generateImage(mood):
    
    moods = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
    
    n_class = moods.index(mood)
    
    # CGAN SECTION
    latent_dim = 128
    noise_and_labels = generate_latent_points(latent_dim, n_class)
    session_gen = onnxruntime.InferenceSession('./models/generator.onnx')
    gen_inputs = session_gen.get_inputs()[0].name 
    img = session_gen.run([], {gen_inputs: noise_and_labels})[0]
    img = img.astype(np.float32)
    

    # UPSCALER SECTION
    if(upscale):
        img = np.transpose(img, (0, 3, 1, 2))
        session_sr = onnxruntime.InferenceSession('./models/RealESRGAN_x2plus.onnx')
        sr_inputs = session_sr.get_inputs()[0].name
        img = session_sr.run([], {sr_inputs: img})[0]

        img = np.clip(img, a_min=0, a_max=1)
        img = np.squeeze(img,axis=0)

        img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))

        img = (img * 255.0).round().astype(np.uint8)

    st.image(img, caption='A {} image!'.format(moods[n_class]), width=500, channels='BGR')
    
    del img, noise_and_labels
    
upscale = st.checkbox('Use Upscaler!', value=True)
    
st.button("Regenerate Image!", on_click=generateImage(mood))




