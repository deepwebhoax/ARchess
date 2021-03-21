import chess
import numpy as np





piece_values = {'P': 10, 'N': 30, 'B': 30, 'R': 50, 'Q': 90, 'K': 100, 'p': -10, 'n': -30, 'b': -30, 'r': -50, 'q': -90, 'k': -100}

# These are all flipped
position_values = {
        'P' : np.array([ [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                        [5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0],
                        [1.0,  1.0,  2.0,  3.0,  3.0,  2.0,  1.0,  1.0],
                        [0.5,  0.5,  1.0,  2.5,  2.5,  1.0,  0.5,  0.5],
                        [0.0,  0.0,  0.0,  2.0,  2.0,  0.0,  0.0,  0.0],
                        [0.5, -0.5, -1.0,  0.0,  0.0, -1.0, -0.5,  0.5],
                        [0.5,  1.0, 1.0,  -2.0, -2.0,  1.0,  1.0,  0.5],
                        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0] ]),

        'N' : np.array([[-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0],
                       [-4.0, -2.0,  0.0,  0.0,  0.0,  0.0, -2.0, -4.0],
                       [-3.0,  0.0,  1.0,  1.5,  1.5,  1.0,  0.0, -3.0],
                       [-3.0,  0.5,  1.5,  2.0,  2.0,  1.5,  0.5, -3.0],
                       [-3.0,  0.0,  1.5,  2.0,  2.0,  1.5,  0.0, -3.0],
                       [-3.0,  0.5,  1.0,  1.5,  1.5,  1.0,  0.5, -3.0],
                       [-4.0, -2.0,  0.0,  0.5,  0.5,  0.0, -2.0, -4.0],
                       [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0] ]),

        'B' : np.array([[-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0],
                       [-1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0],
                       [-1.0,  0.0,  0.5,  1.0,  1.0,  0.5,  0.0, -1.0],
                       [-1.0,  0.5,  0.5,  1.0,  1.0,  0.5,  0.5, -1.0],
                       [-1.0,  0.0,  1.0,  1.0,  1.0,  1.0,  0.0, -1.0],
                       [-1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, -1.0],
                       [-1.0,  0.5,  0.0,  0.0,  0.0,  0.0,  0.5, -1.0],
                       [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0] ]),

        'R' : np.array([[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0],
                       [ 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  0.5],
                       [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                       [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                       [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                       [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                       [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                       [ 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0,  0.0]]),

        'Q' : np.array([[-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0],
                       [-1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0],
                       [-1.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0],
                       [-0.5,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5],
                       [-0.5,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5],
                       [-1.0,  0.5,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0],
                       [-1.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.0, -1.0],
                       [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]]),

        'K' : np.array([[ -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                       [ -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                       [ -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                       [ -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                       [ -2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0],
                       [ -1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0],
                       [  2.0,  2.0,  0.0,  0.0,  0.0,  0.0,  2.0,  2.0 ],
                       [  2.0,  3.0,  1.0,  0.0,  0.0,  1.0,  3.0,  2.0 ]])}


def ndarr_to_board(RealPieces, ndarr):
    boardStr = ''
    for x in ndarr:
        i=0
        while i < len(x):
            if x[i] in RealPieces.keys():
                boardStr+=RealPieces[x[i]]
            else:
                j=1

                while i+1<len(x) and x[i+1] not in RealPieces:
                    j+=1
                    i+=1
                boardStr+=str(j)
            i+=1

        boardStr+='/'
    boardStr=boardStr[:-1]
    board = chess.Board()
    board.set_board_fen(boardStr)
    # board=chess.BaseBoard(boardStr)
    return board



def positionEvaluation(position, piece_values=piece_values, position_values=position_values):
    # Position of pieces is not taken into account for their strength
    if position_values == 'None':
        total_eval = 0
        pieces = list(position.piece_map().values())

        for piece in pieces:
            total_eval += piece_values[str(piece)]

        return total_eval

    else:
        positionTotalEval = 0
        pieces = position.piece_map()

        for j in pieces:
            file = chess.square_file(j)
            rank = chess.square_rank(j)

            piece_type = str(pieces[j])
            positionArray = position_values[piece_type.upper()]

            if piece_type.isupper():
                flippedPositionArray = np.flip(positionArray, axis=0)
                positionTotalEval += piece_values[piece_type] + flippedPositionArray[rank, file]

            else:
                positionTotalEval += piece_values[piece_type] - positionArray[rank, file]

        return positionTotalEval



def minimax(position, depth, alpha, beta, maximizingPlayer, bestMove = 'h1h3'):
    if depth == 0 or position.is_game_over():
        return positionEvaluation(position, piece_values, position_values), bestMove


    if maximizingPlayer:
        maxEval = -np.inf
        for child in [str(i).replace("Move.from_uci(\'", '').replace('\')', '') for i in list(position.legal_moves)]:
            position.push(chess.Move.from_uci(child))
            eval_position = minimax(position, depth-1, alpha, beta, False)[0]
            position.pop()
            maxEval = np.maximum(maxEval, eval_position)
            alpha = np.maximum(alpha, eval_position)
            if beta <= alpha:
                break
        return maxEval

    else:
        minEval = np.inf
        minMove = np.inf
        for child in [str(i).replace("Move.from_uci(\'", '').replace('\')', '') for i in list(position.legal_moves)]:
            position.push(chess.Move.from_uci(child))
            eval_position = minimax(position, depth-1, alpha, beta, True)
            position.pop()
            minEval = np.minimum(minEval, eval_position)
            if minEval < minMove:
                minMove = minEval
                bestMin = child

            beta = np.minimum(beta, eval_position)
            if beta <= alpha:
                break

        return minEval, bestMin

def moveToIndex(move):
    d = {'a':0, 'b':1, 'c':2,'d':3,'e':4, 'f':5, 'g':6, 'h':7 }
    start = (d[move[0]], int(move[1])-1) 
    finish = (d[move[2]], int(move[3])-1)
    return start, finish

RealPieces = {"whitePawn":'P',
    'whiteBishop':'B',
    'whiteKnight':'N',
    'whiteRook':'R',
    'whiteQueen':'Q',
    'whiteKing':'K',
    'blackPawn':'p',
    'blackBishop':'b',
    'blackKnight':'n',
    'blackRook':'r',
    'blackQueen':'q',
    'blackKing':'k'
    'empty':'e'}



def main():
    # board = chess.Board()
    

    ExampleArr=np.array([ ['blackRook', 'blackKnight',  'blackBishop',  'blackQueen',  'blackKing',  'blackBishop',  'blackKnight',  'blackRook'],
                        ['blackPawn', 'blackPawn',  5.0,  'blackPawn',  'blackPawn',  'blackPawn',  'blackPawn',  'blackPawn'],
                        [1.0,  1.0,  'blackPawn',  3.0,  3.0,  2.0,  1.0,  1.0],
                        [0.5,  0.5,  1.0,  2.5,  2.5,  1.0,  0.5,  0.5],
                        [0.0,  0.0,  0.0,  2.0,  2.0,  0.0,  0.0,  0.0],
                        [0.5, -0.5, "whitePawn",  "whitePawn",  0.0, -1.0, -0.5,  0.5],
                        ["whitePawn",  "whitePawn", 0,  0, "whitePawn",  "whitePawn",  "whitePawn",  "whitePawn"],
                        ['whiteRook',  'whiteKnight',  'whiteBishop', 'whiteQueen',  'whiteKing',  'whiteBishop',  'whiteKnight',  'whiteRook'] ])

    board = ndarr_to_board(RealPieces,ExampleArr)
    print(board)
    print(type(np.array(board)))
    # move = minimaxRoot(4,board,True)

    # move = chess.Move.from_uci(str(move))

    # board.push(move)

    minimaxEval, bestMove = minimax(board, 4, -np.inf, np.inf, False)
    print("AI Evaluation: {}\nAI Best Move: {}".format(minimaxEval, bestMove))
    print(type(bestMove))
    move = chess.Move.from_uci(bestMove)
    print(type(move))
    # board.push(move)
    print("{}\n=========================".format(board))
    # print(board)


if __name__ == "__main__":
    main()