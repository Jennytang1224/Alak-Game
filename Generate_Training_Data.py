import pickle

from Alak import Alak
import numpy as np

from Prediction import Prediction

class Generate_Training_data:

    def generate_training_data(self, board, num_of_games):
        # X_data is used for stacking all games, first row set as dummies, need to remove later
        X_data = np.array([[0] * (len(board) * 2)])
        y_data = np.array([])
        j = 0
        user_win_stats = 0
        num_user_starts = 0
        num_computer_starts = 0
        predict = Prediction()
        loaded_model = predict.load_model
        model, type = loaded_model('models/alak_model_v14(85%).h5', 'tf')
        # model, type = loaded_model('models/alak_model_v14.pkl', 'sk')

        for i in range(1, num_of_games + 1):
            game = Alak(board, model, type, user_first=True, interactive=False, random=True, random_start=True, training=False)
            game.pick_starting_piece()
            if game.user_piece == 'x':
                num_user_starts = num_user_starts + 1
            else:
                num_computer_starts = num_computer_starts + 1

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

            # show training data & label for the game
            if not game.is_suicide:  # after the game is done, and if the game is not suicide
                game.embedding()
                print("~~~~~~~~~~~~~~~~~~~~~~~~~ game.X shape: {}".format(np.array(game.X).shape))
                X_data = np.vstack((X_data, np.array(game.X)))
                print("X_data shape: {}".format(X_data.shape))
                # y_data = np.append(y_data, game.y)

        # final training data, label
        X_data = X_data[1:]
        # print(X_data.shape)
        # print(y_data.shape)
        print('user starts: ', num_user_starts, " times")
        print("computer starts: ", num_computer_starts, " times")
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
    X_data, y_data = data.generate_training_data(my_board, 100)
    # data.save_data(X_data, 'data/alak_data_may_15_v1_train.pickle', y_data, 'data/alak_label_may_15_v1_train.pickle')
    #
    # X_data, y_data = data.generate_training_data(my_board, 5000)
    # data.save_data(X_data, 'data/alak_data_may_15_v1_test.pickle', y_data, 'data/alak_label_may_15_v1_test.pickle')
    #
    # X_data, y_data = data.generate_training_data(my_board, 5000)
    # data.save_data(X_data, 'data/alak_data_may_15_v1_val.pickle', y_data, 'data/alak_label_may_15_v1_val.pickle')
