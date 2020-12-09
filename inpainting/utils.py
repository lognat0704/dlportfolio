from PIL import Image
import torch
import numpy as np
from torch.autograd import Variable


class Tool():
    def __init__(self):
        print('Tools loaded')
        
    def tensor_load_rgbimage(self, filename, size=None, scale=None, keep_asp=False):
        img = Image.open(filename).convert('RGB')
        if size is not None:
            if keep_asp:
                size2 = int(size * 1.0 / img.size[0] * img.size[1])
                img = img.resize((size, size2), Image.ANTIALIAS)
            else:
                img = img.resize((size, size), Image.ANTIALIAS)
    
        elif scale is not None:
            img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
        img = np.array(img).transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img
    
    def tensor_save_rgbimage(self, tensor, filename, cuda=True):
        
        #if cuda:
        #    img = tensor.clone().cpu().clamp(0, 255).numpy()
        #else:
        #    img = tensor.clone().clamp(0, 255).numpy()
        img = tensor.clone().cpu().clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype('uint8')
        img = Image.fromarray(img)
        img.save(filename)
        
    def tensor_save_bgrimage(self, tensor, filename, cuda=False):
        (b, g, r) = torch.chunk(tensor, 3)
        tensor = torch.cat((r, g, b))
        self.tensor_save_rgbimage(tensor, filename, cuda)
        
    def preprocess_batch(self, batch):
        batch = batch.transpose(0, 1)
        (r, g, b) = torch.chunk(batch, 3)
        batch = torch.cat((b, g, r))
        batch = batch.transpose(0, 1)
        return batch
    
    def isBase64(self, sb):
        try:
                if type(sb) == str:
                        # If there's any unicode here, an exception will be thrown and the function will return false
                        sb_bytes = bytes(sb, 'ascii')
                elif type(sb) == bytes:
                        sb_bytes = sb
                else:
                        raise ValueError("Argument must be string or bytes")
                return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
        except Exception:
                return False