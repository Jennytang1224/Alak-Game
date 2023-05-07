import pickle

from Alak import Alak
import numpy as np


class Generate_Training_data:

    def generate_training_data(self, board, num_of_games):
        X_data = np.array(
            [[0] * (len(board) * 2)])  # used for stacking all games, first row set as dummies, need to remove later
        y_data = np.array([])
        j = 0
        user_win_stats = 0

        for i in range(1, num_of_games + 1):

            game = Alak(board, interactive=False, random=True)
            game.pick_starting_piece()

            # check if game over
            while game.check_if_game_over() == '_' and len(game.round) != len(game.board) * 2:  # game.round != []:
                game.update_board()
                # record suicide games
                if game.is_suicide:  # and game.round == []:
                    j = j + 1
                    print("********************* number of suicide games: " + str(j))
                    break
                game.switch_piece()

            # calculate suicide stats
            print("********************* number of non-suicide games: " + str(i - j))
            print("****************************************** number of games played: " + str(i))

            # calculate winning stats
            if game.winner == game.user_piece:
                user_win_stats = user_win_stats + 1
                print("__________________user wins:" + str(user_win_stats))

            print("user wins:{:0.0f}% ({} game(s)) and computer wins:{:0.0f}% ({} game(s))".
                  format((user_win_stats / (i - j + .0001)) * 100, user_win_stats,
                         (1 - user_win_stats / (i - j + .0001)) * 100, i - j - user_win_stats))

            # show training data, label for the game
            if not game.is_suicide:  # after the game is done, if the game is not suicide
                game.embedding()
                print("~~~~~~~~~~~~~~~~~~~~~~~~~ game.X shape: {}".format(np.array(game.X).shape))
                # print("yyy:" + str(game.y))
                # print("xxx:" + str(X_data))
                X_data = np.vstack((X_data, np.array(game.X)))
                print("X_data shape: {}".format(X_data.shape))
                y_data = np.append(y_data, game.y)

        # final training data, label
        X_data = X_data[1:]
        # for l in X_data:
        #     print('[' + str(l) + ']\n')
        # print(y_data)
        print(X_data.shape)
        print(y_data.shape)
        return X_data, y_data

    def save_data(self, X_data, X_fname, y_data, y_fname):
        with open(X_fname, 'wb') as handle:
            pickle.dump(X_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(y_fname, 'wb') as handle:
            pickle.dump(y_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("save data to file is completed")


if __name__ == "__main__":
    my_board = 'xxxxx____ooooo'
    data = Generate_Training_data()
    X_data, y_data = data.generate_training_data(my_board, 10000)

    data.save_data(X_data, 'data/alak_data_may_6_v0.pickle', y_data, 'data/alak_label_may_6_v0.pickle')

    # data, label = train.load_data()





