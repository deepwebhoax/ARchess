from chessboard_detection import loadImage, inference
from crop import cropChessboard
from ndar_to_board import ndarr_to_board, RealPieces, minimax, moveToIndex

# from classifier import predict
from graphics import drawMove

from PIL import Image
import numpy as np

def augmentReality(path):
    imGrey = loadImage(path)  # loading for inference function
    im = Image.open(path)     # loading beauty
    
    tiles = inference(imGrey) # getting tiles coordinates
    tilePicturesArray = cropChessboard(np.array(im), tiles)  # getting pictures of tiles for predict
    figureSituation = predict(tilePicturesArray)   # getting figure positions mapped on 8x8
    board = ndarr_to_board(figureSituation)        # changing for conventional datatype    
    minimaxEval, bestMove = minimax(board, 4, -np.inf, np.inf, False) # finding best move
    start, finish = moveToIndex(bestMove)
    move = chess.Move.from_uci(bestMove)
    boardCurrent = board
    board.push(move) # board after best move

    augmentedImage = drawMove(im, tiles, start, finish)
    return augmentedImage

