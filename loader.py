from chessboard_detection import inference, loadImage
from PIL import Image

def prepImage(img_orig):
    img_width, img_height = img_orig.size

    # Resize
    aspect_ratio = min(500.0/img_width, 500.0/img_height)
    new_width, new_height = ((np.array(img_orig.size) * aspect_ratio)).astype(int)
    img = img_orig.resize((new_width,new_height), resample=Image.BILINEAR)
    img = img.convert('L') # grayscale
    img = np.array(img)
    
    return img

def loader(im):
    





def __name__=='__main__':
    im = Image.open()
    im = prepImage(im)
    inference(im)