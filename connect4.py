import numpy as np
import random
import sys
import math
import gym
from gym import spaces
from IPython.display import clear_output

import tensorflow as tf
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.deepq.policies import FeedForwardPolicy

ROW_COUNT = 6
COLUMN_COUNT = 7 
N_CHANNELS = 1

class ConnectFour(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, Player2):
        self.Player2 = Player2
        self.action_space = spaces.Discrete(COLUMN_COUNT)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(ROW_COUNT, COLUMN_COUNT, N_CHANNELS), dtype=np.uint8)

    def reset(self):
        self.board = np.zeros((ROW_COUNT,COLUMN_COUNT)).astype(int)
        self.game_over = False
        self.turn = random.randint(1,2)
        if self.turn == 2:
            state = self.get_state()
            col_p2 = self.Player2.act(state)
            reward = self.make_move(col_p2)
            self.turn = 1
        state = self.get_state()
        return state

    def invalid_move(self, col):
        self.game_over = True
        info = {"message": "GAME OVER. Player {} loses (invalid move at position {}).".format(self.turn, col)}
        if self.turn == 1:
            reward = -1
        else:
            reward = 1
        return reward, info

    def winning_move(self, col):
        self.game_over = True
        info = {"message": "GAME OVER. Player {} wins (winning move at position {}).".format(self.turn, col)}
        if self.turn == 1:
            reward = 1
        else:
            reward = -1
        return reward, info
    
    def tie_move(self, col):
        self.game_over = True
        info = {"message": "GAME OVER. Player {} tied the game (final move at position {}).".format(self.turn, col)}
        reward = 0
        return reward, info

    def valid_move(self, col):
        self.drop_piece(col)
        if self.check_winning_move():
            reward, info = self.winning_move(col)
        elif not self.moves_left(): 
            reward, info = self.tie_move(col)
        else:
            reward = 0
            info = ""
        return reward, info

    def drop_piece(self, col):
        for row in range(ROW_COUNT):
            if self.board[row][col] == 0:
                break
        self.board[row][col] = self.turn

    def is_valid_location(self, col):
        return (self.board[ROW_COUNT-1][col] == 0)

    def make_move(self, col):
        assert col in np.arange(COLUMN_COUNT), "{} is an invalid move".format(col)
        valid = self.is_valid_location(col)
        if not valid:
            reward, info = self.invalid_move(col)
        else: 
            reward, info = self.valid_move(col)
        return reward, info

    def step(self, col_p1):        
        reward, info = self.make_move(col_p1)
        if self.game_over:
            state = self.get_state()
        else: 
            self.turn = (self.turn % 2) + 1
            state = self.get_state()
            col_p2 = self.Player2.act(state)
            reward, info = self.make_move(col_p2)
            self.turn = (self.turn % 2) + 1
        state = self.get_state()
        return state, reward, self.game_over, info

    def get_state(self, for_render=False):
        if self.turn == 1:
            state = np.where(self.board==2, -1, self.board)
        if self.turn == 2:
            state = np.where(self.board==2, 1, -self.board)
        state = np.flip(state, 0)
        if not for_render:
            state = np.expand_dims(state, axis=2)
        return state

    def render(self, mode='human'):
        outfile = sys.stdout
        state = self.get_state(True)
        outfile.write('| 0 | 1 | 2 | 3 | 4 | 5 | 6 |\n')
        outfile.write('+===+===+===+===+===+===+===+\n')
        for s in range(ROW_COUNT*COLUMN_COUNT):
            position = np.unravel_index(s, state.shape)
            if state[position] == 1:
                output = "|" + "\x1b[3;31;41m" + " 1 " + "\x1b[0m"
            elif state[position] == -1:
                output = "|" + "\x1b[3;33;43m" + " 2 " + "\x1b[0m"
            else:
                output = "|   "
            if (position[1] + 1) % COLUMN_COUNT == 0:
                output += '|\n+---+---+---+---+---+---+---+\n'
            outfile.write(output)
        outfile.write('\n')

    def check_winning_move(self):
        piece = self.turn
        # horizontal win
        for c in range(COLUMN_COUNT-3):
            for r in range(ROW_COUNT):
                if self.board[r][c] == piece and self.board[r][c+1] == piece and self.board[r][c+2] == piece and self.board[r][c+3] == piece:
                    return True
        # vertical win
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT-3):
                if self.board[r][c] == piece and self.board[r+1][c] == piece and self.board[r+2][c] == piece and self.board[r+3][c] == piece:
                    return True
        # positive diagonal win
        for c in range(COLUMN_COUNT-3):
            for r in range(ROW_COUNT-3):
                if self.board[r][c] == piece and self.board[r+1][c+1] == piece and self.board[r+2][c+2] == piece and self.board[r+3][c+3] == piece:
                    return True
        # negative diagonal win
        for c in range(COLUMN_COUNT-3):
            for r in range(3, ROW_COUNT):
                if self.board[r][c] == piece and self.board[r-1][c+1] == piece and self.board[r-2][c+2] == piece and self.board[r-3][c+3] == piece:
                    return True
                
    def moves_left(self):
        return any([self.is_valid_location(i) for i in np.arange(COLUMN_COUNT)])
    

class Manual_Player():
    def __init__(self):
        pass
    def act(self, state, env):
        clear_output()
        env.render()
        col = None
        while col is None:
            print("Select a column [0-6]:")
            input_value = input()
            try: 
                col = int(input_value)
            except ValueError:
                print("{} is not a valid column.".format(input_value))
        return col

    
def play_agent(Player2):
    Player1 = Manual_Player()
    env = ConnectFour(Player2)
    state = env.reset()
    while True:
        env.render()
        action = Player1.act(state, env)
        state, reward, done, info = env.step(action)
        if done:
            clear_output()
            env.render()
            print(info["message"])
            break
            
def modified_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = conv_to_fc(layer_2)
    return activ(linear(layer_2, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

class CustomCnnPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomCnnPolicy, self).__init__(*args, **kwargs, cnn_extractor=modified_cnn, feature_extraction="cnn")
        