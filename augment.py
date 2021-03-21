from chessboard_detection import loadImage, inference
from crop import cropChessboard
from ndar_to_board import ExampleArr, ndarr_to_board, RealPieces, minimax, moveToIndex

# from classifier import predict
from graphics import drawMove, drawTiles

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import chess

def augmentReality(path):
    imGrey = loadImage(path)  # loading for inference function
    im = Image.open(path)     # loading beauty
    
    tiles = inference(imGrey) # getting tiles coordinates (64x)
    plt.imshow(im)
    plt.plot(tiles[:,])
    tilePicturesArray = cropChessboard(np.array(im), tiles)  # getting pictures of tiles for predict

    # figureSituation = predict(tilePicturesArray)   # getting figure positions mapped on 8x8
    figureSituation = ExampleArr
    board = ndarr_to_board(RealPieces,figureSituation)        # changing for conventional datatype    
    minimaxEval, bestMove = minimax(board, 4, -np.inf, np.inf, False) # finding best move
    start, finish = moveToIndex(bestMove)
    move = chess.Move.from_uci(bestMove)
    boardCurrent = board
    board.push(move) # board after best move
    
    # augmentedImage = drawMove(np.array(im), tiles, start, finish)
    augmentedImage = drawTiles(np.array(im), tiles)
    return Image.fromarray(augmentedImage)

