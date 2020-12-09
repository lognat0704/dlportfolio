import os
import time
from PIL import Image
from PIL.ImageOps import grayscale
from watchdog.events import RegexMatchingEventHandler
import os 

class ImagesEventHandler(RegexMatchingEventHandler):
    THUMBNAIL_SIZE = (128, 128)
    IMAGES_REGEX = [r".*[^_thumbnail]\.jpg$"]

    def __init__(self):
        super().__init__(self.IMAGES_REGEX)

    def on_created(self, event):
        file_size = -1
        while file_size != os.path.getsize(event.src_path):
            file_size = os.path.getsize(event.src_path)
            time.sleep(1)
                     
        if 'BACKGROUND' in event.src_path:
            self.process(event)

    def process(self, event):
        cp_result = 1 ## 1 - not successful, 0 - successful
        face_result = 1
        
        try:
            while(cp_result != 0):
                cp_result = os.system('cp '+event.src_path+' /home/nelson/facefactory/showcase/canvas.png')
                if cp_result == 0: ## cp successful
                    print('cp success')
                    break
                else:
                    print('will try to cp again')
                    time.sleep(0.1)
                        
            while(face_result != 0):
                face_result = os.system('python face_entertain.py --input_img /home/nelson/facefactory/showcase/canvas.png --output_path /home/nelson/facefactory/showcase/')
                if face_result == 0:
                    break
                else:
                    print('will try to face again')
                    time.sleep(5)
            
            
                    
        except KeyboardInterrupt:
            print('interrupted!')
        
            
        
        
        