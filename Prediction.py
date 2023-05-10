import tensorflow as tf
import numpy as np


class Prediction:

    def load_model(self, model_fname):
        new_model = tf.keras.models.load_model(model_fname)
        print(new_model.summary())
        return new_model

    # take in the board after opponent moves,
    # return all possible boards after moves
    def generate_successor(self, cur_board, cur_piece):
        # loop through board and find '_'
        successor_boards = {}  # {successor_board: (from, to)}
        move_from_lst = []
        move_to_lst = []
        for i, piece in enumerate(cur_board):  # ox_xx____o___ , cur = o
            if piece == cur_piece:
                move_from_lst.append(i)
            if piece == '_':
                move_to_lst.append(i)
        print(move_to_lst)
        print(move_from_lst)
        for x in move_from_lst:
            for y in move_to_lst:
                arr = np.array(list(cur_board))
                arr[x] = '_'
                arr[y] = cur_piece
                new_board = "".join(arr)
                successor_boards[new_board] = (x, y)
                print(new_board)
        print("Number of successor boards: " + str(len(successor_boards)))

        return successor_boards

    # use model to predict each successor and get probabilities
    def predict(self, successor_boards, model, board):
        embedded_board = np.array(self.embedding(board))

        X_data = np.array([[0] * len(board) * 2])
        print(X_data.shape)

        for b, move in successor_boards.items():
            embedded_successor = np.array(self.embedding(b))
            row = np.concatenate((embedded_board, embedded_successor), axis=0)
            X_data = np.vstack((X_data, row))

        y_pred_prob = model.predict(X_data)
        print(y_pred_prob)
        idx = np.argmax(np.array(y_pred_prob), axis=0)
        best_move_prob = y_pred_prob[idx[0]]
        best_move = list(successor_boards.values())[idx[0]]

        return best_move, best_move_prob

    def embedding(self, board): # convert board to -1, 0 and 1s
        embedded = []
        for c in board:
            if c == 'x':
                embedded.append(-1)
            elif c == '_':
                embedded.append(0)
            else:
                embedded.append(1)
        return embedded


if __name__ == '__main__':
    p = Prediction()
    model = p.load_model('models/alak_model_v1.h5')
    board = 'oooxx_xxoxx_x_'
    user_piece = 'o'
    successors = p.generate_successor(board, user_piece)
    optimal_move, optimal_move_probability = p.predict(successors, model, board)
    print("best_move is from {} to {} and best_move_prob:{}".format(optimal_move[0], optimal_move[1], optimal_move_probability[0]))
