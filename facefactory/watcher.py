import sys
import time
import os

from watchdog.observers.inotify import InotifyObserver as Observer
from watchdog.events import FileSystemEventHandler

import threading

class ImagesWatcher:
    def __init__(self, src_path):
        self.__src_path = src_path
        
        self.__event_handler = ImagesEventHandler()
        
        #self.__event_observer = PollingObserver()
        self.__event_observer = Observer()

    def run(self):
        ## display the welcome image
        os.system('cp /home/nelson/facefactory/showcase/canvas_w.png /home/nelson/facefactory/showcase/canvas.png')
        
        self.__event_observer.schedule(self.__event_handler, self.__src_path, recursive=True)
        self.__event_observer.start()
        print('Paparazzi is on the way')
        
        try:
            while True:
                ## if the image on screen (either the welcome image or the face image) is displayed more than 15 minutes, 
                ## replace it with the welcome image. 
                ## This is to prevent the face image being displayed too long
                if(time.time() - os.path.getmtime('showcase/canvas.png') >= 15 * 60):
                    os.system('cp /home/nelson/facefactory/showcase/canvas_w.png /home/nelson/facefactory/showcase/canvas.png')
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.__event_observer.stop()
            ## display the offline image
            os.system('cp /home/nelson/facefactory/showcase/canvas_o.png /home/nelson/facefactory/showcase/canvas.png')
            
        self.__event_observer.join()

## disply output from style transfer, change every 15 seconds, until face factory output is ready
def show_style_transfer_results(path, face_output_path):
    style_transfer_files = os.listdir(path)
    i = 0
    
    while not os.path.isfile(os.path.join(face_output_path, 'canvas.png')) : ## if face part not finished yet
        ## show one style image and sleep 15 seconds
        os.system('cp ' + os.path.join(path, style_transfer_files[i]) +' /home/nelson/facefactory/showcase/canvas.png')
        time.sleep(15)
        i += 1
        i = i % len(style_transfer_files)
    

class ImagesEventHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()

    def on_created(self, event):
        if not event.is_directory and 'FACE_SNAP' in event.src_path: ## if new backgroupd file created
            try:
                ## if the image on screen (most likely previous face image) is displayed less than 15 seconds, wait, do nothing
                while (time.time() - os.path.getmtime('showcase/canvas.png') < 15):
                       time.sleep(1)
                
                ## display image from camera on screen
                while(os.system('cp '+ event.src_path + ' /home/nelson/facefactory/showcase/canvas.png') != 0):
                    print('will try to cp again')
                    time.sleep(0.1)
                    
                ## create style transfer images
                while(os.system('python /home/nelson/aibrush/brush_entertain.py --input_img ' + event.src_path + ' --output_path output') != 0):
                    print('will try style transfer again!')
                    time.sleep(5)
                    
                ## create and display face image in a new folder under showcase
                base = os.path.basename(event.src_path)
                output_dir = os.path.join('showcase', os.path.splitext(base)[0])
                print(output_dir)
                os.makedirs(output_dir, exist_ok=True)
                
                t = threading.Thread(target=show_style_transfer_results, args=('/home/nelson/aibrush/single_style/', output_dir))
                t.setDaemon(True)  # 设置为守护线程
                t.start()
                
        
                while(os.system('python face_entertain.py --input_img ' + event.src_path + ' --output_path ' + output_dir) != 0):
                    print('will try to do face again')
                    time.sleep(5)
                
                while(os.system('cp '+output_dir+'/canvas.png'+ ' /home/nelson/facefactory/showcase/canvas.png') != 0):
                    print('will try to cp again')
                    time.sleep(0.1)
                
                #while (time.time() - os.path.getmtime(output_dir+'/canvas.png') < 15):
                #       time.sleep(1)
                
                    
            except KeyboardInterrupt:
                print('interrupted!')


if __name__ == "__main__":
    src_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    ImagesWatcher(src_path).run()