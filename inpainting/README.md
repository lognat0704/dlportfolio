# Image Inpainting

<img src="http://www.hmtseng.com/inpainting.png">

### Project Description
Develop an image in-painting model for image editing. Able to remove logo and unwanted object

### Responsibility
-Train the Inpaint Contextual Attention Model using CelebA-HQ dataset

-Build its Flask App and host on Tencent Cloud Market

* inpainting_run.py : input original image and its mask to produce the image with the indicated object

* api.py : Build in-painting Flak Restful API for Tencent cloud market
