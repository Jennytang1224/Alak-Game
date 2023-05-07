from Alak import Alak
from colorama import Fore, Back, Style

class Test:

    def test_remove(self, verbose=False):
        """
        The tests include, for both sides:
        - simple kill
        - double kill
        - double kill that involves more than one piece
        - suicide moves.
        """

        board_list = ['xoxoxx________', 'xooxooxx______',
                      '__xoo__oxo____', '__xoooox_o____',
                      '_o__ooxxxo__o_', '___xoxxxoo____', '___x_oxxoo_____',
                      '____xoxx___x___']
        board_expect = ['x_x_xx________', 'x__x__xx______',
                        '__xoo__o_o____', '__x____x_o____',
                        '_o__oo___o__o_', '___xo___oo____', '___x_o__oo_____',
                        '____x_xx___x___']
        off_side = ['x', 'x', 'x', 'x', 'o', 'o', 'o', 'o']
        pos = [2, 3, 8, 2, 5, 4, 5, 5]
        fail_num = 0

        print("Testing test_remove after move from offensive side:")
        for i, b in enumerate(board_list):
            print("\nAfter {:s} move to {}: ".format(off_side[i], pos[i]))

            # set board to be the way I want
            game = Alak(b, interactive=False, random=True)
            if verbose:
                print("Test {:d}".format(i))
                print('Board: ', game.board)

            # I check normal capture vs. suicide capture in 2 different fucntions:
            # checking normal capture
            game.cur_piece = off_side[i]
            game.check_capture(pos[i], off_side[i])

            # checking suicide capture
            if game.board == b:
                game.cur_piece = game.find_opponent(off_side[i])
                game.check_suicide_capture(pos[i], game.find_opponent(off_side[i]))
                game.print_board()

            try:
                assert game.board == board_expect[i]
            except:
                print('i', i)
                fail_num += 1

        tot_num = len(board_list)
        print('{:d} out of {:d} tests passed'.format(len(board_list) - fail_num, len(board_list)))
        if fail_num > 0:
            print(Fore.RED + '{:d} out of {:d} tests failed'.format(fail_num, len(board_list)))

if __name__ == "__main__":
    test = Test()
    test.test_remove()