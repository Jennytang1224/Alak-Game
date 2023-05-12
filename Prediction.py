import pickle

import tensorflow as tf
import numpy as np


class Prediction:

    def load_model(self, model_fname, type):
        if type == 'tf':
            new_model = tf.keras.models.load_model(model_fname)
            # print(new_model.summary())
        else:
            with open(model_fname, 'rb') as f:
                new_model = pickle.load(f)
        return new_model, type

    # take in the board after opponent moves,
    # return all possible boards after moves
    def generate_successor(self, cur_board, cur_piece):
        successor_boards = {}  # {successor_board: (from, to)}
        move_from_lst = []
        move_to_lst = []
        for i, piece in enumerate(cur_board):  # ox_xx____o___ , cur = o
            if piece == cur_piece:
                move_from_lst.append(i)
            if piece == '_':
                move_to_lst.append(i)
        # print(move_to_lst)
        # print(move_from_lst)
        for x in move_from_lst:
            for y in move_to_lst:
                arr = np.array(list(cur_board))
                arr[x] = '_'
                arr[y] = cur_piece
                new_board = "".join(arr)
                successor_boards[new_board] = (x, y)
                # print(new_board)
        # print("Number of successor boards: " + str(len(successor_boards)))
        return successor_boards

    # use model to predict each successor and get probabilities
    def predict(self, successor_boards, model, board, type, cur_piece):
        embedded_board = np.array(self.embedding(board, cur_piece))
        X_data = np.array([[0] * len(board) * 2])

        for b, move in successor_boards.items():
            embedded_successor = np.array(self.embedding(b, cur_piece))
            row = np.concatenate((embedded_board, embedded_successor), axis=0)
            X_data = np.vstack((X_data, row))
        X_data = X_data[1:]
        if type == 'sk':
            y_pred_prob = model.predict_proba(X_data)
            print(y_pred_prob)
        else: # 'tf'
            y_pred_prob = model.predict(X_data)
            print(y_pred_prob)

        idx = np.argmax(np.array(y_pred_prob), axis=0)
        print(successor_boards.values())
        print("idx:", str(idx))

        if type == 'sk': # for sklearn model
            best_move_prob = y_pred_prob[idx[1]]
            best_move = list(successor_boards.values())[idx[0]]
            print("best_move is from {} to {} and best_move_prob:{}"
                  .format(best_move[0], best_move[1], best_move_prob[0]))
            # if model generate suicide move, find the next best move with highest prob
            if self.is_suicide_move(best_move[0], best_move[1], cur_piece, board):
                print("update best move")
                temp = y_pred_prob.tolist()
                print(temp)
                temp.pop(idx[0])
                print(temp)
                temp_board = list(successor_boards.values())
                del temp_board[idx[0]]
                print(temp_board)
                idx = np.argmax(np.array(temp), axis=0)
                print(idx)
                best_move_prob = temp[idx[0]]
                best_move = temp_board[idx[0]]

        else: # for tensorflow model, idx is a tuple
            best_move_prob = y_pred_prob[idx[0]]
            best_move = list(successor_boards.values())[idx[0]]

            #  if model generate suicide move, find the next best move with highest prob
            if self.is_suicide_move(best_move[0], best_move[1], cur_piece, board):
                print("update best move")
                temp = y_pred_prob
                temp = np.delete(temp, idx)
                temp_board = list(successor_boards.values())
                del temp_board[idx[0]]
                idx = np.argmax(np.array(temp), axis=0)
                best_move_prob = temp[idx]
                best_move = temp_board[idx]

        print("best_move is from {} to {} and best_move_prob:{}"
              .format(best_move[0], best_move[1], best_move_prob))
        return best_move, best_move_prob

    def embedding(self, board, cur_piece): # convert board to -1, 0 and 1s
        embedded = []
        for c in board:
            if c == cur_piece:
                embedded.append(1)
            elif c == '_':
                embedded.append(0)
            else:
                embedded.append(-1)
        return embedded

    def find_opponent(self, piece_to_check):
        if piece_to_check == 'o':
            return 'x'
        else:
            return 'o'

    def is_suicide_move(self, src, dest, piece_to_check, b):
        arr = np.array(list(b))
        arr[src] = '_'
        arr[dest] = piece_to_check
        board = "".join(arr)

        is_suicide = False
        opp_piece = self.find_opponent(piece_to_check)
        left_capture_counter = 0
        right_capture_counter = 0

        # check to the left
        pos = dest
        if pos == 0 or pos == len(board) - 1:
            return is_suicide

        while pos > 0:
            pos -= 1
            if board[pos] == '_':
                return is_suicide
            elif board[pos] == piece_to_check:
                left_capture_counter += 1
                if pos == 0:
                    return is_suicide
            elif board[pos] == opp_piece:  # find opposite, continue look fo more
                break

        # check to the right
        pos = dest
        while pos < len(board) - 1:
            pos += 1
            if board[pos] == '_':
                return is_suicide
            elif board[pos] == piece_to_check:
                right_capture_counter += 1
                if pos == len(board) - 1:
                    return is_suicide

            elif board[pos] == opp_piece:  # find opposite, continue look fo more
                is_suicide = True
                return is_suicide

        return is_suicide





if __name__ == '__main__':
    p = Prediction()
    model, type = p.load_model('models/alak_model_v10.h5', 'tf')
    board = 'oooxx_xxoxx_x_'
    piece = 'o'
    successors = p.generate_successor(board, piece)
    optimal_move, optimal_move_probability = p.predict(successors, model, board, type, piece)
    print("best_move is from {} to {} and best_move_prob:{}"
          .format(optimal_move[0], optimal_move[1], optimal_move_probability[0]))
