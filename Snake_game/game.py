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

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    # Tại sao phải truyền Enum vào class này ??
    # Tránh nhầm lẫn với số 
    # Sẽ dùng 
    # direction == Direction.RIGHT thay vì 
    # direction == 3: 

Point = namedtuple("Point", "x, y") 
# namedtuple là một hàm trong collections giúp tạo ra một 
# class bất biến (immutable) giống như tuple nhưng có tên thuộc tính
# Để có thể viết như sau : --------- p = Point(10, 20)
# ---------------------------------- print(p.x)  -> 10
# ---------------------------------- print(p.y)  -> 20


# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 1000000000000

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        # Lưu kích thước màn hình 
        self.w=w
        self.h=h 
        # Init display 
        self.display = pygame.display.set_mode((self.w, self.h))      # Tạo cửa sổ UI
        pygame.display.set_caption('Snake')                           # Đặt tên cửa sổ 
        self.clock = pygame.time.Clock()
        self.reset()                                                  # Gọi reset để khởi tạo trạng thái game
    def reset(self):
        # init game state 
        self.direction = Direction.RIGHT                              # Khi game bắt đầu, hướng con rắn luôn hướng sang phải 

        self.head = Point(
            self.w / 2,                                               # Đầu con rắn sẽ nằm ngay chính giữa map
            self.h / 2
        )
 
        self.snake = [                                                # Danh sách cách đoạn thân của con rắn
            self.head,                                                # Khúc đầu 
            Point(self.head.x - BLOCK_SIZE, self.head.y),             # Khúc giữa, con rắn ban đầu nằm ngang, tọa độ x khúc giữa cách tọa độ x của Khúc đầu 1 đơn vị (BLOCK_SIZE), tọa độ y bằng nhau vì nó nằm ngang
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)        # Khúc cuối, con rắn ban đầu nằm ngang, tọa độ y khúc cuối cách tọa độ x của Khúc đầu 2 đơn vị (2 * BLOCK_SIZE), tọa độ y bằng nhau vì nó nằm ngang
        ]

        self.score = 0                                                # Điểm ban đầu bằng 0
        self._place_food()                                            # Gọi hàm khởi tạo thức ăn
        self.frame_iteration = 0                                      # Mỗi lần gọi play_step() thì frame_teration += 1 
    

    def _place_food(self):                                            # Hàm khởi tạo vị trí food 
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE 

        self.food = Point(x, y)                                       # Vị trí food 
        if self.food in self.snake:
            self._place_food()
        # Nếu khởi tạo food nằm trên các điểm của con rắn, phải tạo lại một vị trí khác 
    def play_step(self, action):
        self.frame_iteration += 1                                     # Qua 1 play_step thì frame_iteration tăng lên 1  
        
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move 
        self._move(action)                                            # Tính hướng mới và cập nhật self.head (tọa độ mới của đầu) dựa trên action và self.direction 
        self.snake.insert(0, self.head)                               # Chèn vị trí đầu mới vào đầu danh sách các điểm của con rắn 

        # Check if game over 
        reward = 0
        game_over = False 

        if self.is_collision() == True or self.frame_iteration > 100*len(self.snake):
            game_over = True 
            reward = -10                                              # Đâm vào tường thì bị trừ reward thôi 
            return reward, game_over, self.score 
        

        # Place new food or just move 
        if self.head == self.food: 
            self.score += 1
            reward = 10 

            self._place_food()                                        # Ăn xong thì phải tạo ra một food mới 

            return reward, game_over, self.score 
        else: 
            self.snake.pop()                                          # Khi di chuyển bình thường, xóa phần tử cuối ở đuổi 
        

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return reward, game_over, self.score
    def is_collision(self, pt=None):                                           # Kiểm tra va chạm với tường hay không 
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.y > self.h - BLOCK_SIZE or pt.x < 0 or pt.y < 0:
            return True 
        
        if pt in self.snake[1:]:                               # Kiểm tra đâm chính nó 
            return True 
        
        return False 
    
    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):    
        # Các tùy chọn di chuyển 
        # [straight, right, left]

        # Tạo một danh sách các hướng thứ tự theo chiều kim đồng hồ : Phải -> Xuống > Trái -> Lên
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP] 

        # Tìm vị trí index của hướng hiên tại 
        idx = clock_wise.index(self.direction)

        # Di chuyển 
        if np.array_equal(action, [1, 0, 0]):                          # Đi thẳng 
            new_way = self.direction
        
        elif np.array_equal(action, [0, 1, 0]):                        # Right turn 
            new_idx = (idx + 1) % 4
            new_way = clock_wise[new_idx]
        elif np.array_equal(action, [0, 0, 1]):                      # LEft turn 
            new_idx = (idx - 1) % 4 
            new_way = clock_wise[new_idx]
        
        # Gán lại vào self.direction hướng mới sau khi chọn các option move 
        self.direction = new_way

        x = self.head.x 
        y = self.head.y 

        # Cập nhật vị trí đầu rắn 

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE 
        elif self.direction == Direction.DOWN: 
            y -= BLOCK_SIZE
        else:
            y += BLOCK_SIZE 

        self.head = Point(x,y) 
         


          



