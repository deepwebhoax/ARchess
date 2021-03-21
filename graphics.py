from skimage.draw import line_aa, circle
import numpy as np

def getPosCoord(tiles, boardPos):  # returns coordinates of the center of a tile given index(8x8)
    tiles = np.reshape(tiles, (8,8,4))
    coords = tiles[boardPos[0], boardPos[1]]
    return abs(int((coords[0]-coords[1])/2.)), abs(int((coords[2]-coords[3])/2.))

def drawMove(im, tiles, start, finish):
    im = np.array(im)
    c1, c2 = getPosCoord(tiles, start), getPosCoord(tiles, finish)
    rr, cc, val = line_aa(c1[0],c1[1], c2[0], c2[1])
    im[rr, cc] = val * 255
    return im

def drawTiles(im, tiles):
    im = np.array(im)
    print(tiles)
    for tile in tiles:
        
        rr, cc = circle(int(tile[0]), int(tile[2]), 4)
        im[rr, cc] = 100
        rr, cc = circle(int(tile[1]), int(tile[3]), 4)
        im[rr, cc] = 100
    
    return im


