import cv2
from PIL import Image, ImageDraw
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from mtcnn import MTCNN

detector = MTCNN()

def prep_mtcnn(pic):
    image = Image.fromarray((pic).astype(np.uint8))
    image = image.resize((224, 224))
    face_array = np.asarray(image)
    return face_array
def extract_faces(pic):
    pic = prep_mtcnn(pic)    
    # detect faces in the image
    results = detector.detect_faces(pic)
    return results
def plot_res_mtcnn(pic, faces):
    fig, ax = plt.subplots()
    ax.imshow(pic)
    for face in faces:
        length = face['box'][3] - face['box'][1]
        width = face['box'][2] - face['box'][0]
        
        # key points
        for key, val in face['keypoints'].items():
            ax.scatter(val[0],  val[1])
        face_points = [i[1] for i in face['keypoints'].items()]
        face_points.append((face['box'][0],face['box'][1]))
        face_points.append((face['box'][2],face['box'][3]))
        # face box
        x1, y1 = min(face_points, key=lambda x:x[0])[0], min(face_points, key=lambda x:x[1])[1]
        x2, y2 = max(face_points, key=lambda x:x[0])[0], max(face_points, key=lambda x:x[1])[1]
        ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none'))
        # ax.scatter(face['box'][0], face['box'][1])
        # ax.scatter(face['box'][2], face['box'][3])

def restore_original(coords, orig):
    print(orig)
    width_ratio = 224./orig[0]
    height_ratio= 224./orig[1]
    return (int(coords[0]/height_ratio), int(coords[1]/width_ratio))

def draw_rectangles(image, rectangles, color=[0,255,255]):
    """
    image - numpy image - RGB format
    rectangles - (x1, y1, x2, y2) rectangle coords
            or a list of (x1, y1, x2, y2)
    returns changed image
    """
    print(type(rectangles))

    if not (type(rectangles) is list):
        rectangles = [rectangles]
    print(rectangles)
    for rectangle in rectangles:
        for i in range(rectangle[1], min(rectangle[3], image.shape[0])):
            for j in range(rectangle[0], min(rectangle[2], image.shape[1])):
                image[i][j] = color
    return image


def mark_faces_mtcnn(im):
    """
    im - np.array
    returns im with drawn rectangles (np.array)
    """
    # old draw
    # if type(im)!="<class 'numpy.ndarray'>":
    # imarr = Image.fromarray(im)
    # print(type(imarr))
    # draw = ImageDraw.Draw(imarr)
    original_size = im.shape
    faces = extract_faces(im)

    rectangles = []
    for face in faces:
        face_points = [i[1] for i in face['keypoints'].items()]
        # face_points.append((face['box'][0],face['box'][1]))
        # face_points.append((face['box'][2],face['box'][3]))
        # squeezed face boxes 
        x1, y1 = min(face_points, key=lambda x:x[0])[0], min(face_points, key=lambda x:x[1])[1]
        x2, y2 = max(face_points, key=lambda x:x[0])[0], max(face_points, key=lambda x:x[1])[1]
        x1, y1 = restore_original((x1,y1), original_size)
        x2, y2 = restore_original((x2,y2), original_size)
        rectangles.append((x1,y1,x2,y2))
        
        # draw.rectangle((x1, y1, x2, y2), fill=(0, 192, 192), outline=(255, 255, 255))
    print(original_size)
    print(rectangles)
    im_changed = im
    # im_changed = draw_rectangles(im, rectangles)
    # plt.imshow(im_changed)
    # plt.show()
    return im_changed



if __name__=='__main__':
    mark_faces_mtcnn(np.array(Image.open('peeps1.png')))