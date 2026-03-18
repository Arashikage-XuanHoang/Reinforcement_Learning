import pygame 
import numpy as np
import random 
from enum import Enum 
from collections import namedtuple 
import requests 
from io import BytesIO

pygame.init()

# BƯỚC 1: Link đúng (phải là raw.githubusercontent.com, không phải github.com/blob)
url = "https://raw.githubusercontent.com/patrickloeber/snake-ai-pytorch/main/arial.ttf"

# BƯỚC 2: Tải font về dưới dạng bytes
response = requests.get(url)
response.raise_for_status()  # nếu link sai sẽ báo lỗi ngay
font_bytes = BytesIO(response.content)

font = pygame.font.Font(font_bytes, 25)
Block_size = 20
Speed = 20

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)


class Direction(Enum):
    Right = 1 
    Left = 2 
    Up = 3 
    Down = 4

class Point:
    def __init__(self, x, y):
        self.x = x 
        self.y = y

class SnakeGameAI: 
    def __init__(self, w = 640, h = 480):
        self.w = w 
        self.h = h 

        # Init display 
        self.display = pygame.display.set_mode((self.w, self.h))      # Tạo cửa sổ UI
        pygame.display.set_caption('Snake')                           # Đặt tên cửa sổ 
        self.clock = pygame.time.Clock()
        reward = 0 
        game_over = False 
        self.reset() 
    def reset(self):
        self.direction = Direction.Right                              # Khởi tạo hướng đi ban đầu - luôn sang phải 
        self.score = 0 
        

        x = self.w / 2 
        y = self.h / 2

        self.head = Point(x, y)
        self.snake = [
            self.head
        ]
        self._play_food()                                             # Khởi tạo vị trí của food 
        self.frame_iteration = 0                                      # Số lần dịch chuyển 1 Block_size của đầu rắn 
    def _play_food(self):
        x = random.randint(0, (self.w - Block_size) // Block_size) * Block_size 
        y = random.randint(0, (self.h - Block_size) // Block_size) * Block_size 

        food_location = Point(x, y) 

        # --- Nếu thức ăn trùng với thân rắn - thiết lập lại --- 
        if food_location in self.snake: 
            self._play_food()       
        self.food_loca = food_location

    def play_step(self, action):
        self.reward = 0

        self.game_over = False
        self.frame_iteration += 1    

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # 1.1 move 
        self._move(action)

        # 2. Check game_over 
        if self.va_cham():
            self.reward = -10
            self.game_over = True 
            return self.game_over, self.reward, self.score 
        
        # 2.1 Nếu ăn được food : reward = 10 
        elif self.head == self.food_loca:
            self.reward = 10
            self.score += 1 
            
            # --- Tạo thức ăn mới --- 
            self._play_food()  

            return self.game_over, self.reward, self.score 
        # 2.2. Nếu không ăn được food 
        else:
            self.snake.pop()    # Xóa đuổi 

        # 3. Update clock and ui 
        self._update_ui()
        self.clock.tick(Speed)

        # 4. return game over and score
        return self.game_over, self.reward, self.score
    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, Block_size, Block_size))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food_loca.x, self.food_loca.y, Block_size, Block_size))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


        
        



    def va_cham(self, pt=None):
        if pt is None: 
            pt = self.head 
        
        # --- Đâm vào tường --- 
        if pt.x < 0 or pt.x > self.w - Block_size or pt.y < 0 or pt.y > self.h - Block_size: 
            return True 
        
        # --- Đâm vào bản thân --- 
        if pt in self.snake[1:]:
            return True 
        return False 
               

    def _move(self, action):

        # ---- Tạo hướng ---  
        clock_wise = [
            Direction.Right,
            Direction.Down,
            Direction.Left,
            Direction.Up

        ]

        # --- Tìm vị trí idx trong clock_wise của hướng hiện tại --- 
        idx = clock_wise.index(self.direction) 

        # --- Di chuyển --- 
        # --- Di chuyển thẳng [1, 0, 0] --- Di chuyển phải [0, 1, 0] --- Di chuyển trái [0, 0, 1] ---  

        # Di chuyển sang phải - action == [1, 0, 0]
        if np.array_equal(action, [1, 0, 0]):
            new_way = self.direction 

        # Di chuyển sang trái
        elif np.array_equal(action, [0, 0, 1]):
            idx = (idx - 1) % 4 
            new_way = clock_wise[idx]
        # Di chuyển sang phải 
        else: 
            idx = (idx + 1) % 4 
            new_way = clock_wise[idx]

        self.direction = new_way 


        # Cập nhật vị trí đầu 
        x = self.head.x 
        y = self.head.y 

        if self.direction == Direction.Right:
            x += Block_size 
        elif self.direction == Direction.Left:
            x -= Block_size 
        elif self.direction == Direction.Down: 
            y -= Block_size 
        else: 
            y += Block_size 
        
        # --- Cập nhật vị trí đầu --- 
        self.head = Point(x, y) 

        # --- Cập nhật đầu rắn danh sách --- 

        self.snake.insert(0, self.head)       # Cập nhật đầu 


 




    


    