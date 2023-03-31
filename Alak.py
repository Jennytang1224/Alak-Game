import numpy as np
import sys


class Alak:
    def __init__(self):
        self.board = ""
        self.prev_dest = -1
        self.cur_piece = ''
        self.step_counter = 0

    # generate a starting board with length n, and default pieces
    def generate_board(self):
        self.board = 'xxxx__oxoo'
        print("board starts with: " + self.board)
        return np.array(list(self.board))

    def pick_starting_piece(self):
        self.cur_piece = np.random.choice(['x', 'o'])
        print("'" + self.cur_piece + "' starts the game:")
        return self.cur_piece

    # go through the board and return indices of all given slots
    def find_all_specific_slots(self, slot):
        arr = np.array(list(self.board))
        return np.where(arr == slot)[0]

    # move the piece from src to dest, print the move and increase the step_counter
    def move_piece(self, src, dest):
        self.step_counter += 1
        print("-> Step " + str(self.step_counter) + ": move '" + self.cur_piece + "' from position "
              + str(src) + " to position " + str(dest))

        arr = np.array(list(self.board))
        arr[src] = '_'
        arr[dest] = self.cur_piece
        self.board = "".join(arr)

    # given the current piece, go from the source(src) to the destination(dest)
    def update_board(self):
        # current rule: randomly pick a spot of current piece
        # and move to a random empty spot


        # check prev dest create any captures (previous suicide move)
        if self.prev_dest != -1:
            # if self.cur_piece == 'o':
            self.check_capture(self.prev_dest, self.cur_piece)
            # else:
            #     self.check_capture(self.prev_dest, self.find_opponent('o'))
            print("board after capture: ", self.board)

        list_of_empty_slots = self.find_all_specific_slots('_')
        list_of_cur_piece_slots = self.find_all_specific_slots(self.cur_piece)
        #
        # # check if the empty slots are in 'KO' condition, if so, we shouldn't consider them as potential dest
        # list_of_valid_empty_slots = self.clean_KO_slots(list_of_empty_slots)

        # randomly pick moves but later will model and predict the move
        src = np.random.choice(list_of_cur_piece_slots)
        print("empty slots: ", list_of_empty_slots)
        dest = np.random.choice(list_of_empty_slots)

        # move the piece to the dest
        self.move_piece(src, dest)
        print("board after move: ", self.board)

        # check capture in current board
        self.check_capture(dest, self.cur_piece)
        print("board after captures: ", self.board)
        self.prev_dest = dest



    # check if there's any captures:
    # 1. after put down the piece, check both left and right side if the first one is the opposite piece
    # 2. if so, keep going til find the same piece, then we capture the opposite pieces in between
    def check_capture(self, dest, piece_to_check):
        opp_piece = self.find_opponent(piece_to_check)
        left_capture_counter = 0
        right_capture_counter = 0

        # check to the left
        pos = dest
        while pos > 0:
            pos -= 1
            if self.board[pos] == '_':
                left_capture_counter = 0 # nothing captured, continue the game
                self.capture(dest, left_capture_counter, 'left')
                break
            elif self.board[pos] == self.cur_piece:
                self.capture(dest, left_capture_counter, 'left')
                break
            elif self.board[pos] == opp_piece: # find opposite, continue look fo more
                left_capture_counter += 1

        # check to the right
        pos = dest
        while pos < len(board)-1:
            pos += 1
            if self.board[pos] == '_':
                right_capture_counter = 0  # nothing captured, continue the game
                self.capture(dest, left_capture_counter, 'right')
                break
            elif self.board[pos] == self.cur_piece:
                self.capture(dest, right_capture_counter, 'right')
                break
            elif self.board[pos] == opp_piece:  # find opposite, continue look fo more
                right_capture_counter += 1

    # discard pieces from the board based on the counter and direction
    def capture(self, start_from, capture_counter, direction):
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
            print("board after capture:", self.board)
        else:
            print("No capture :( ")

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
        # game ends when only one side of the piece exists on board
        if len(set(self.board)) == 2:  # ended
            winner = [elem for elem in set(self.board) if elem != '_'][0]
            print("GAME OVER!! Winner is '" + winner + "'")
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    game = Alak()
    board = game.generate_board()
    starting_piece = game.pick_starting_piece()

    # check if game over
    while game.check_if_game_over() == '_':
        print("")
        print("***********************")

        game.update_board()
        game.switch_piece()

    sys.exit()

    # made a move

    # check board on any captures

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
