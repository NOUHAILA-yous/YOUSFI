!pip install deepface opencv-python-headless
# Capture d'image dans Colab
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import cv2
import numpy as np

def take_photo(filename='photo.jpg', quality=0.8):
    js = Javascript('''
        async function takePhoto(quality) {
            const div = document.createElement('div');
            const video = document.createElement('video');
            const stream = await navigator.mediaDevices.getUserMedia({video: true});

            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            // Wait for user to press a button
            await new Promise(resolve => {
                const button = document.createElement('button');
                button.textContent = 'Prendre une photo';
                div.appendChild(button);
                button.onclick = () => resolve();
            });

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);

            stream.getTracks().forEach(track => track.stop());
            div.remove();

            return canvas.toDataURL('image/jpeg', quality);
        }
    ''')
    display(js)
    data = eval_js('takePhoto({})'.format(quality))
    binary = b64decode(data.split(',')[1])

    with open(filename, 'wb') as f:
        f.write(binary)

    return filename

# Prendre une photo
photo_path = take_photo()

from deepface import DeepFace
import matplotlib.pyplot as plt

# Charger l'image
img = cv2.imread(photo_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Analyse
result = DeepFace.analyze(img_rgb, actions=['emotion'], enforce_detection=False)

# Affichage du résultat
plt.imshow(img_rgb)
plt.axis('off')
plt.title("Emotion détectée: " + result[0]['dominant_emotion'])
plt.show()
