# Artificial Neural Network Alak Player

The purpose of this project is to build an artificial neural network (ANN) algorithm to play the
one-dimensional version of the ancient board game Go, or Alak.
Alak: https://senseis.xmp.net/?Alak

## Project Steps Overview:
1. Based on the rules of the game, simulate the game so the game can play random games interactive or non-interactively (Alak.py)
2. Use program generated randomly played games as training data: (Generate_Training_Data.py)
        
        1). attributes: 
            save each round of the board and concatenate both user and computer's side of the board after their move, then use embedding to convert the boards only contains 1, 0, -1: 1(user's piece), 0(empty), -1(opponent's piece)
        2). target: 
            a list of 1 or -1, if user won the whole game, for each round of the game, the corresponding labels these rounds will be 1, otherwise -1 (this way is to optimize the final win or lose, rather than amount of kills in each round)
3. Create either TensorFlow NN model or Sklearn MLPClassifier model, save models (Models.py)
4. To test model: Load models and use it to play against 100 random games to see the winning rate. model wins > 60 % means the model has certain level of intelligence. My final model's winning rate against 100 random games is ~85%.



## Rules to play the game:

1. There are two sides: the “x” side and the “o” side.
2. Starting board configuration: there are 5 pieces lined up on either side and 14 spaces total, empty spaces between the two groups. ie: xxxxx____ooooo
3. You can move one piece at a time.
4. You cannot add any new pieces to the board.
5. The “x” side always makes the first move.
6. Kill: if and only if one or more contiguous pieces on one side is “sandwiched” between two
opposing pieces, then the “sandwiched” piece(s) will be removed from the board.



## Simulations:
1. Show the positions of the pieces in each round of simulation (0 to 13), by using 0-9 and lower case ‘abcd’. Here’s the opening round of a simulated game:

        Starting Game:
        Your side is 'o'
        round: 0
        Board: xxxxx____ooooo
            0123456789abcd

        x : 0 ==> 5
        Board: _xxxxx___ooooo
            0123456789abcd
        gain: 0

        o : 9 ==> 0
        Board: oxxxxx___o_ooo
            0123456789abcd
        gain: 0
2. When a move results in a kill, show the board immediately after the move and again show the board after the killed pieces are removed, as shown in this example:

            o : 7 ==> 4
            Board: _xxxoxo_oo
                0123456789
            Board: _xxxo_o_oo
                0123456789
            gain: 1
3. Repeat until one side wins (the other side should have <= 1 piece on the board)

4. There are different ways to use the code: 

The Alak game contains following parameters: (Inside the Alak())

- board: initial board to start the game
- model: loaded trained model 
- type: model type, either 'sk' as Sklearn model or 'tf' as TensorFlow model
- user_first: if True, user will be assigned to 'x' and always goes first; otherwise, user will be always assigned as 'o' and goes second 
- interactive : if True, human play with computer; else computer play with computer
- random : if True, computer play random moves; else computer play moves generated by model
- random_start: if True, randomly select the first player (x); else depends on user_first parameter 
- training: if True, games play for training data; else games won't be used for training data

a. To play 1 game interactively with the trained model: (user input + model)
in Alak.py:

                my_board = 'xxxxx____ooooo'
                predict = Prediction()
                loaded_model = predict.load_model
                model, type = loaded_model('models/alak_model_v14(83%).h5', 'tf')
                game = Alak(my_board, model, type, user_first=False, interactive=True, random=False, random_start=True, training=False)
                game.play_1_game()
                
b. To play 1 game interactively with random moves: (user_input + random)
in Alak.py:    

                my_board = 'xxxxx____ooooo'
                game = Alak(my_board, model, type, user_first=False, interactive=True, random=True, random_start=True, training=False)
                game.play_1_game()
              
c. To create training data from random games with random starts, user_first doesnt matter if random_start = True (model + random)
in Generate_Traing_Data: 

                my_board = 'xxxxx____ooooo'
                game = Alak(my_board, model, type, user_first=False, interactive=False, random=True, random_start=True, training=True)
               
             
d. To create training data from the games played by model agaist model with random starts, user_first doesnt matter if random_start = True (model + model)
in Generate_Traing_Data:               
                
                my_board = 'xxxxx____ooooo'
                game = Alak(my_board, model, type, user_first=False, interactive=False, random=False, random_start=True, training=True)
              
                
e. To test how good the trained model is by playing against random moves(model + ranodm)
in Generate_Traing_Data:

                my_board = 'xxxxx____ooooo'
                game = Alak(board, model, type, user_first=True, interactive=False, random=True, random_start=True, training=False)


## Training:
1. In my training data, I only allow user suicide, model suicide will trigger the whole game get discarded, and won't appear in the training data
2. I also detect any illigal move from user input: 

• out of range
• not from 0-9 or a-d
• move from and move to are the same slots
• try to move opponent's piece
• move from or move to slots have other pieces on them
3. When training ended, it will show the stats: 
example:

        !!!! GAME OVER!! Winner is 'x' 
        ********************* number of non-suicide games: 100
        ****************************************** number of games played: 100
        user wins:14% (14 game(s)) and computer wins:86% (86 game(s))
        user starts:  48 times
        computer starts:  52 times

## Models:
1. You can call either TensorFlow NN model or Sklearn MLPClassifier model
3. After the model was trained, I save all the rounds (each round consists of two moves, one by each side) of the simulated game in a pickle file if using sklearn, or h5 if using TensorFlow.

## Prediction:
1. load model
2. generate all successor boards from the previous move
3. feed the successor boards in to trained model, and model outputs the probability of winning corresponding to each successor board.
4. in the prediction stage, check if the highest probability suggests a suicide move. If so, I checked the next highest probability move; repeat until a non-suicide move is selected. 

## Test:
In Test.py, I tested:

• simple kill involving different numbers of pieces being removed
• double kill
• double kill that involves more than one piece




## Future work:
1. To optimize the number of pieces killed by the move, I may try using a
softmax to label the different number of kills, instead of using all the same 1 or -1 to label the whole game

2. I used random games to train: This would be similar to the AlphaGoZero training sample; In this
case, I could also train the game twice: first generate the games randomly and
train; and then use your trained classifier to play lots of games and then use these to train again.
