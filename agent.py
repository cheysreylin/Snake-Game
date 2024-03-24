import torch
import random
import numpy as np
from collections import deque # deque: datastructure where we want to store the memory 
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000 
LR = 0.001 # LR = LEARNING RATE

class Agent:
    def __init__(self):
        self.number_game = 0
        self.epsilon = 0 # control the randomness
        self.gamma = 0 # discount rate
        self.memory = deque(maxlen= MAX_MEMORY) # if the exist the memory, will automaticly remove element from the left -> popleft()
        self.model = None 
        self.trainer = None 
        # model, trainer 


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x, # food left 
            game.food.x > game.head.x, # food right 
            game.food.y < game.head.y, # food up 
            game.food.y > game.head.y # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done): # done or game over
        self.memory.append((state, action, reward, next_state, done)) # popleft if the max_memory is reached


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # return a list of tuples
        else:
            mini_sample = random.sample.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action,  reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.number_game
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

        

# global function
def train():
    plot_scores = [] # create empty to list to keep track of the score, and use for plotting 
    plot_mean_score = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get the old stae
        state_old = agent.get_state(game)

        # get move of current state
        final_move = agent.get_action(state_old)

        # perform the move and get new state
        reward, done, score = game.play_step(final_move) 
        state_new = agent.get_state(game)

        # train the short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember and store it in memory
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train the long memory plot the result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_score.append(mean_score)
            plot(plot_scores, plot_mean_score)


if __name__ == '__main__':
    train()