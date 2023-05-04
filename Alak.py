import numpy as np
import sys


class Alak:
    def __init__(self):
        self.board = ""
        self.prev_dest = -1
        self.cur_piece = ''
        self.round_counter = 0
        self.user_piece = ''
        self.computer_piece = ''
        self.X = []
        self.y = []
        self.round =[]

    # generate a starting board with length n, and default pieces
    def generate_board(self):
        self.board = 'xxxxx____ooooo'
        # print("\nInitial board: " + self.board)
        self.print_board()
        return np.array(list(self.board))

    def pick_starting_piece(self):
        # system automatically assign user a piece:
        self.user_piece = np.random.choice(['x', 'o'])
        if self.user_piece == 'x':  # user goes first
            self.computer_piece = 'o'
            print("Your side is 'x'")
        else:
            self.computer_piece = 'x'
            print("Your side is 'o'")

        self.cur_piece = 'x'  # x always goes first
        print("\n~~~~~~~~~~~~~~~ " + self.cur_piece + " starts the game: ~~~~~~~~~~~~~~~~")

    # go through the board and return indices of all given slots
    def find_all_specific_slots(self, slot):
        arr = np.array(list(self.board))
        return np.where(arr == slot)[0]

    # move the piece from src to dest, print the move and increase the step_counter
    def move_piece(self, src, dest):
        if self.cur_piece == 'x':
            self.round_counter += 1
            print("-> Round " + str(self.round_counter))

        print("Move " + self.cur_piece + " from position " + str(src) + " to position " + str(dest))
        arr = np.array(list(self.board))
        arr[src] = '_'
        arr[dest] = self.cur_piece
        self.board = "".join(arr)

    # given the current piece, go from the source(src) to the destination(dest)
    def update_board(self):
        # check if prev dest create any captures (previous suicide move)
        if self.prev_dest != -1:
            # if self.cur_piece == 'o':
            self.check_capture_from_last_round(self.prev_dest, self.cur_piece)
            # print("clean board: ", self.board)
            print("clean board:")
            self.print_board()
            # save moves after embedding
            self.embedding()

        print("\n~~~~~~~~~~~~~~~ " + self.cur_piece + "'s turn: ~~~~~~~~~~~~~~~~")

        if self.user_piece == self.cur_piece:  # use's turn: use input prompt
            print("Please enter your move below:")
            from_pos = input("Move from position:")
            to_pos = input("To position:")
            print("You are moving {} to {}".format(from_pos, to_pos))

            while not from_pos.isnumeric() or not to_pos.isnumeric() \
                    or int(from_pos) > len(self.board) - 1 or int(from_pos) < 0 \
                    or self.board[int(from_pos)] != self.user_piece \
                    or int(to_pos) > len(self.board) or int(to_pos) < 0 \
                    or self.board[int(to_pos)] != '_' \
                    or from_pos == to_pos:
                print("This is an illegal move, please re-enter your move again:")
                from_pos = input("Move from position:")
                to_pos = input("To position:")
                print("You are moving {} to {}".format(from_pos, to_pos))

            src = int(from_pos)
            dest = int(to_pos)

        else:  # computer's turn:  randomly make move
            list_of_empty_slots = self.find_all_specific_slots('_')
            list_of_cur_piece_slots = self.find_all_specific_slots(self.cur_piece)
            #
            # # check if the empty slots are in 'KO' condition, if so, we shouldn't consider them as potential dest
            # list_of_valid_empty_slots = self.clean_KO_slots(list_of_empty_slots)

            # randomly pick moves but later will model and predict the move
            src = np.random.choice(list_of_cur_piece_slots)
            # print("empty slots: ", list_of_empty_slots)
            dest = np.random.choice(list_of_empty_slots)

        # move the piece to the dest
        self.move_piece(src, dest)
        # print("board after move: ", self.board)
        print("board after move:")
        self.print_board()

        # check capture in current board
        self.check_capture(dest, self.cur_piece)
        # print("board after capture: ", self.board)
        print("board after capture:")
        self.print_board()
        self.prev_dest = dest

    # check if there's any captures:
    # 1. after put down the piece, check both left and right side if the first one is the opposite piece
    # 2. if so, keep going til find the same piece, then we capture the opposite pieces in between
    def check_capture(self, dest, piece_to_check):
        opp_piece = self.find_opponent(piece_to_check)
        left_capture_counter = 0
        right_capture_counter = 0
        no_capture_flag_left = True
        no_capture_flag_right = True

        # check to the left
        pos = dest
        while pos > 0:
            pos -= 1
            if self.board[pos] == '_':
                left_capture_counter = 0  # nothing captured, continue the game
                self.capture(dest, left_capture_counter, 'left')
                break
            elif self.board[pos] == self.cur_piece:
                self.capture(dest, left_capture_counter, 'left')
                no_capture_flag_left = False
                break
            elif self.board[pos] == opp_piece:  # find opposite, continue look fo more
                left_capture_counter += 1

        # check to the right
        pos = dest
        while pos < len(board) - 1:
            pos += 1
            if self.board[pos] == '_':
                right_capture_counter = 0  # nothing captured, continue the game
                self.capture(dest, right_capture_counter, 'right')
                break
            elif self.board[pos] == self.cur_piece:
                self.capture(dest, right_capture_counter, 'right')
                no_capture_flag_right = False
                break
            elif self.board[pos] == opp_piece:  # find opposite, continue look fo more
                right_capture_counter += 1

        if no_capture_flag_left is False and no_capture_flag_right is False:
            print(self.cur_piece + " captured " + str(
                left_capture_counter + right_capture_counter) + " pieces! => DOUBLE KILL")

        if no_capture_flag_left:
            left_capture_counter = 0
        if no_capture_flag_right:
            right_capture_counter = 0

        if no_capture_flag_left or no_capture_flag_right:
            print(self.cur_piece + " captured " + str(left_capture_counter + right_capture_counter) + " pieces!")

    # discard pieces from the board based on the counter and direction
    def capture(self, start_from, capture_counter, direction):
        num_piece_captured = capture_counter
        if capture_counter != 0:
            while capture_counter > 0:
                if direction == 'left':
                    start_from -= 1
                    capture_counter -= 1
                    self.board = self.board[0: start_from] + '_' + self.board[start_from + 1: len(self.board)]
                if direction == 'right':
                    start_from += 1
                    capture_counter -= 1
                    self.board = self.board[0: start_from] + '_' + self.board[start_from + 1: len(self.board)]

    def check_capture_from_last_round(self, dest, piece_to_check):
        opp_piece = self.find_opponent(piece_to_check)
        left_capture_counter = 0
        right_capture_counter = 0

        # check to the left
        pos = dest
        while pos > 0:
            pos -= 1
            if self.board[pos] == '_':
                left_capture_counter = 0  # nothing captured, continue the game
                break
            elif self.board[pos] == self.cur_piece:
                left_capture_counter += 1
                break
            elif self.board[pos] == opp_piece:  # find opposite, continue look fo more
                left_capture_counter += 1

        # check to the right
        pos = dest
        while pos < len(board) - 1:
            pos += 1
            if self.board[pos] == '_':
                right_capture_counter = 0  # nothing captured, continue the game
                break
            elif self.board[pos] == self.cur_piece:
                right_capture_counter += 1
                break
            elif self.board[pos] == opp_piece:  # find opposite, continue look fo more
                right_capture_counter += 1

        if left_capture_counter > 0 and right_capture_counter > 0:  # capture in between
            self.capture(dest + 1, left_capture_counter, 'left')
            self.capture(dest - 1, right_capture_counter, 'right')
            print("Suicide move alert: " + self.cur_piece + " captured " + str(
                left_capture_counter + right_capture_counter - 1) + " piece(s)!")
            print("!!! Detected suicide move, have to start a new game! ")
            exit(1)

    # # given a list of potential destinations, return the valid slots can be destinations without 'KO' condition
    # def clean_KO_slots(self, list_of_empty_spots):
    #     opp = self.find_opponent()
    #
    #     list_of_valid_empty_slots = []
    #     print(list_of_empty_spots)
    #     for slot in list_of_empty_spots:
    #         # print(self.board[slot-1], self.board[slot+1])
    #         if (slot - 1 < 0) or (slot + 1 > len(self.board) - 1) \
    #                 or (self.board[slot-1] == opp and self.board[slot+1] == opp):
    #             continue
    #         else:
    #             list_of_valid_empty_slots.append(slot)
    #     print(np.array(list_of_valid_empty_slots))
    #     return np.array(list_of_valid_empty_slots)

    # check if the game ends with a winner
    def check_if_game_over(self):
        # game ends when only one side has 1 piece left
        count_x = self.board.count('x')
        count_o = self.board.count('o')
        if count_x <= 1:
            winner = 'o'
            print("\n!!!! GAME OVER!! Winner is '" + winner + "'")
            self.label_round()
            return winner
        elif count_o <= 1:
            winner = 'x'
            print("\n!!!! GAME OVER!! Winner is '" + winner + "'")
            self.label_round()
            return winner
        else:  # not ended
            return '_'

    def train(self):
        return 0

    def switch_piece(self):
        self.cur_piece = self.find_opponent(self.cur_piece)

    def find_opponent(self, piece_to_check):
        if piece_to_check == 'o':
            return 'x'
        else:
            return 'o'

    def print_board(self):
        expanded_board = ' '.join(self.board)
        print(expanded_board)
        print("0 1 2 3 4 5 6 7 8 9 10111213\n")

    def embedding(self):  # x side + o side
        if len(self.round) < len(self.board) * 2:
            for c in self.board:
                if c == 'x':
                    self.round.append(-1)
                elif c == '_':
                    self.round.append(0)
                else:
                    self.round.append(1)
            if len(self.round) == len(self.board) * 2:
                self.X.append(self.round)  # append current full round
                # for l in self.X:
                #     print('[' + str(l) + ']\n')
                self.round = []  # reset the list

    def label_round(self):
        print(self.check_if_game_over)
        if self.check_if_game_over == self.user_piece: # user is the winner
            self.y = [0] * self.round_counter
        else:
            self.y = [1] * self.round_counter


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    game = Alak()
    board = game.generate_board()
    game.pick_starting_piece()
    # check if game over
    while game.check_if_game_over() == '_':
        game.update_board()
        game.switch_piece()

    # print(game.X)
    game.embedding()
    for l in game.X:
        print('[' + str(l) + ']\n')
    print(game.y)

    sys.exit()
