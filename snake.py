import random
import typing as t
from collections import namedtuple
from enum import Enum

import numpy as np
import pygame

pygame.init()
font = pygame.font.Font("arial.ttf", 25)
# font = pygame.font.SysFont('arial', 25)

Action = tuple[int, int, int]


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

# rgb colors
WHITE: t.Final[pygame.Color] = pygame.Color(255, 255, 255)
RED: t.Final[pygame.Color] = pygame.Color(200, 0, 0)
BLUE1: t.Final[pygame.Color] = pygame.Color(0, 0, 255)
BLUE2: t.Final[pygame.Color] = pygame.Color(0, 100, 255)
BLACK: t.Final[pygame.Color] = pygame.Color(0, 0, 0)

BLOCK_SIZE: t.Final[int] = 20
SPEED: t.Final[int] = 100000000
TIMEOUT = 100


class SnakeGameAI:
    def __init__(self, w: int = 640, h: int = 480) -> None:
        self.w: int = w
        self.h: int = h

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()

        # init game state
        self.reset()

    def reset(self) -> None:
        self.direction: Direction = Direction.RIGHT

        self.head: Point = Point(self.w / 2, self.h / 2)
        self.snake: list[Point] = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]

        self.score: int = 0
        self.food: Point = self._place_food()
        self.frame_iteration = 0

    def _place_food(self) -> Point:
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

        self.food = (point := Point(x, y))
        if self.food in self.snake:
            self._place_food()
        return point

    def play_step(self, action: Action) -> tuple[int, bool, int]:
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self.frame_iteration += 1
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > TIMEOUT * len(self.snake):
            game_over = True
            reward = -10
            print("Timeout | ", end="")
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt: t.Optional[Point] = None) -> bool:
        # hits boundary
        if pt is None:
            pt = self.head
        if (
            pt.x > self.w - BLOCK_SIZE
            or pt.x < 0
            or pt.y > self.h - BLOCK_SIZE
            or pt.y < 0
        ):
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self) -> None:
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(
                self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )
            pygame.draw.rect(
                self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12)
            )

        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action: Action) -> None:
        # [straight, right, left]

        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        if np.array_equal(action, [0, 1, 0]):
            new_dir = clockwise[(idx + 1) % 4]
        elif np.array_equal(action, [0, 0, 1]):
            new_dir = clockwise[(idx - 1) % 4]
        else:
            new_dir = self.direction

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
