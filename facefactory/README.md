 # Facefactory

<img src="http://www.hmtseng.com/face.png">

### Project Description
A asian face generator creates face image and replace original face in short video for marketing. The face generator helps a single short video can be duplicated into many alike videos by replacing the model face for massive viral marketing

### Responsibility
-Develop Asia face generator utilizing EfficientNet based encoder and StyleGAN. Generate indicated Asian face without retraining StyleGAN model

-Develop related Flask app and scripts for taking face image from camera  and display the generated face images mixed with indicated latents (smile, gender, age) on TV.

* face_entertain.py: generate face images mixed with latent directions (age, smile, gender)

* watcher.py: monitor the camera, and take face image when a person steps in. Run face_entertain.py to generate face images, and display them on TV

* panel.ipynb: tryout all functions separately - Aligned_images, images to latent, latent optimize, latent mixer, latent to image