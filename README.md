# MECGAN - Mood to Emotional-Conditional Generative Adversarial Network

## Summary
Find in this repo all the notebooks and code used to develop our music to Emotion to GAN flow for CIS4914. We sought to create a gan that could generate emotional artwork given some class labels. The class labels for the GANs were originally obtained by passing in a set of Spotify songs into an ANN and classifying them as one of 8 moods. The complete product can be found at [here](https://mixage.me/) with complete Spotify integration. However, in this repo contains the code exclusive to the development of the GAN. You can find the GAN hosted seperately in streamlit [here](https://share.streamlit.io/jmho/mecgan/main/app.py)

## How it works
1. To develop this we first trained a CGAN using the ArtEmis dataset of labeled emotional art. This was done in TensorFlow
2. Then, we acquired a pretrained 2x REAL-ESRGAN from the projects repo.
3. Next we converted both models to ONNX runtime to speed up the process and to shrink the file sizes.
4. Finally we attatched everything together as seen in app.py

## Acknowledgement
This work was not for profit and was for our own interest to see what we could make given a semester. All this is to mention, none of our work would be possible without the work of the following individuals 

`@article{achlioptas2021artemis,
    title={ArtEmis: Affective Language for Visual Art},
    author={Achlioptas, Panos and Ovsjanikov, Maks and Haydarov, Kilichbek and
            Elhoseiny, Mohamed and Guibas, Leonidas},
    journal = {CoRR},
    volume = {abs/2101.07396},
    year={2021}
}`

`https://keras.io/examples/generative/conditional_gan/`

`https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/`

`https://github.com/xinntao/Real-ESRGAN`
