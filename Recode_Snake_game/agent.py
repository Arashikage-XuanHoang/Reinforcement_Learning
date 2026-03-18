import torch 
import random 
import numpy as np 
from collections import deque 
from game import SnakeGameAI, Direction, Point 
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
Block_size = 20 
LR = 0.001 

def sat_tuong(game, point):
    w = game.w 
    h = game.h 

    if point.x < 0 or point.x > w or point.y < 0 or point.y > h:  # Sát ngay tường
        return True  
    if point in game.snake[1:]:                                   # Sát ngay thân
        return True 
    return False 


class Agent: 

    def __init__(self):
        self.n_games = 0                        # Số game đã trải qua 
        self.epsilon = 0                        # randomness 
        self.gamma = 0                          # discount rate 
        self.model = Linear_QNet(11, 256, 3)
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)              

    def get_state(self, game):
        head = game.head

        point_l = Point(head.x - Block_size, head.y)           # Tọa độ điểm -- trái -- của đâu rắn
        point_r = Point(head.x + Block_size, head.y)           # Tọa độ điểm -- phải -- của đầu rắn
        point_u = Point(head.x, head.y + Block_size)           # Tọa độ điểm -- trên -- của đầu rắn
        point_d = Point(head.x, head.y - Block_size)           # Tọa độ điểm -- dưới -- của đầu rắn


        # state = [
        # --- danger straight, danger right, danger left ---
        # --- direction left, direction right, direction up, direction down --- 
        # --- food left, food right, food up, food down --- 
        # ]

        # --- hàm is_collision(pt): có đối số là pt một điểm 

        dir_right = game.direction == Direction.Right 
        dir_left = game.direction == Direction.Left 
        dir_up = game.direction == Direction.Up
        dir_down = game.direction == Direction.Down 

        danger_straight = 0 # --- Có tường hoặc thân trước mắt --- 
        if (sat_tuong(game, point_l) and dir_left) or (dir_right and sat_tuong(game, point_r)) or (dir_up and sat_tuong(game, point_u)) or (dir_down and sat_tuong(game, point_d)):
            danger_straight = 1 
        
        danger_left = 0     # --- Có tường hoặc thân bên trái mắt --- 
        if (dir_right and sat_tuong(game, point_u)) or (dir_left and sat_tuong(game, point_d)) or (dir_up and sat_tuong(game, point_l)) or (dir_down and sat_tuong(game, point_r)):
            danger_left = 1 

        danger_right = 0    # --- Có tường hoặc thân bên phải mắt --- 
        if (dir_right and sat_tuong(game, point_d)) or (dir_left and sat_tuong(game, point_u)) or (dir_up and sat_tuong(game, point_r)) or (dir_down and sat_tuong(game, point_l)): 
            danger_right = 1 

        food_left = 0 
        food_right = 0 
        food_up = 0 
        food_down = 0 

        if game.food_loca.x > head.x:
            food_right = 1 
        else:
            food_left = 1 
        
        if game.food_loca.y > head.y: 
            food_up = 1 
        else:
            food_down = 1 
        
        state = [
            danger_straight, danger_right, danger_left, 
            dir_left, dir_right, dir_up, dir_down, 
            food_left, food_right, food_up, food_down
        ]
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE: 
            mini_sample = random.sample(self.memory, BATCH_SIZE)      
        else:
            mini_sample = self.memory 

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done): 
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):  # --- Sự lựa chọn giữa exploitation và exploration 
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        move = random.randint(0, 2)
        if random.randint(0, 200) < self.epsilon: 
            final_move[move] = 1 
        else: 
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0 
    record = 0 
    agent = Agent()
    game = SnakeGameAI() 
    while True:
        # Get old state 
        state_old = agent.get_state(game)

        # Get move 
        final_move = agent.get_action(state_old)

        # Performance of old_state and get new state 
        done, reward, score = game.play_step(final_move) 

        next_state = agent.get_state(game)

        # remember 
        agent.remember(state_old, final_move, reward, next_state, done)

        if done:   # khi trò chơi kết thúc -- huấn luyện mô hình 
            game.reset() 
            agent.n_games +=  1 
            agent.train_long_memory()

            if score > record:
                record = score 
                agent.model.save()
            
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()



                                                   


        