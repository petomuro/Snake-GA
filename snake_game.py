import pygame
import time
import random

# INITIALIZE
pygame.init()


class SnakeGame:
    def __init__(self):
        # GAME OVER
        self.GAME_CLOSE = False

        # COLORS
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.BLACK = (0, 0, 0)

        # DISPLAY
        self.DISPLAY_WIDHT = 400
        self.DISPLAY_HEIGHT = 400
        self.DISPLAY = pygame.display.set_mode(
            (self.DISPLAY_WIDHT, self.DISPLAY_HEIGHT))
        pygame.display.set_caption('Snake')

        # FPS
        self.CLOCK = pygame.time.Clock()

        # FONT
        self.FONT_STYLE = pygame.font.SysFont(None, 24)

        # SNAKE PROPERTIES
        self.SNAKE_BLOCK = 40
        self.SNAKE_SPEED = 999999

        # MOVEMENT
        self.SNAKEX = self.DISPLAY_WIDHT / 2
        self.SNAKEY = self.DISPLAY_HEIGHT / 2

        # CHANGE MOVE
        self.SNAKEX_CHANGE = 0
        self.SNAKEY_CHANGE = 0

        # SNAKE
        self.SNAKE_LIST = []

        self.LENGHT_OF_SNAKE = 3
        self.VERTICAL = random.randint(0, 1)
        self.COUNT = 0

        for _ in range(self.LENGHT_OF_SNAKE):
            if self.VERTICAL == 0:
                SNAKE_SEGMENT = [self.SNAKEX - self.COUNT, self.SNAKEY]
            else:
                SNAKE_SEGMENT = [self.SNAKEX, self.SNAKEY - self.COUNT]

            self.SNAKE_LIST.insert(0, SNAKE_SEGMENT)
            self.COUNT += self.SNAKE_BLOCK

        # FOOD
        self.FOODX = round(random.randrange(
            0, self.DISPLAY_WIDHT - self.SNAKE_BLOCK) / self.SNAKE_BLOCK) * self.SNAKE_BLOCK
        self.FOODY = round(random.randrange(
            0, self.DISPLAY_HEIGHT - self.SNAKE_BLOCK) / self.SNAKE_BLOCK) * self.SNAKE_BLOCK

        self.CHECK_COORDINATES = True

        while (self.CHECK_COORDINATES):
            for i in self.SNAKE_LIST:
                if i == [self.FOODX, self.FOODY]:
                    self.FOODX = round(random.randrange(
                        0, self.DISPLAY_WIDHT - self.SNAKE_BLOCK) / self.SNAKE_BLOCK) * self.SNAKE_BLOCK
                    self.FOODY = round(random.randrange(
                        0, self.DISPLAY_HEIGHT - self.SNAKE_BLOCK) / self.SNAKE_BLOCK) * self.SNAKE_BLOCK
                else:
                    self.CHECK_COORDINATES = False

        # MAX STEPS
        self.MAX_STEPS = 100

        # RECORD
        self.RECORD = self.LENGHT_OF_SNAKE - 1

    def reset(self):
        # GAME OVER
        self.GAME_CLOSE = False

        # MOVEMENT
        self.SNAKEX = self.DISPLAY_WIDHT / 2
        self.SNAKEY = self.DISPLAY_HEIGHT / 2

        # SNAKE
        self.SNAKE_LIST = []

        self.LENGHT_OF_SNAKE = 3
        self.VERTICAL = random.randint(0, 1)
        self.COUNT = 0

        for _ in range(self.LENGHT_OF_SNAKE):
            if self.VERTICAL == 0:
                SNAKE_SEGMENT = [self.SNAKEX - self.COUNT, self.SNAKEY]
            else:
                SNAKE_SEGMENT = [self.SNAKEX, self.SNAKEY - self.COUNT]

            self.SNAKE_LIST.insert(0, SNAKE_SEGMENT)
            self.COUNT += self.SNAKE_BLOCK

        # FOOD
        self.FOODX = round(random.randrange(
            0, self.DISPLAY_WIDHT - self.SNAKE_BLOCK) / self.SNAKE_BLOCK) * self.SNAKE_BLOCK
        self.FOODY = round(random.randrange(
            0, self.DISPLAY_HEIGHT - self.SNAKE_BLOCK) / self.SNAKE_BLOCK) * self.SNAKE_BLOCK

        self.CHECK_COORDINATES = True

        while (self.CHECK_COORDINATES):
            for i in self.SNAKE_LIST:
                if i == [self.FOODX, self.FOODY]:
                    self.FOODX = round(random.randrange(
                        0, self.DISPLAY_WIDHT - self.SNAKE_BLOCK) / self.SNAKE_BLOCK) * self.SNAKE_BLOCK
                    self.FOODY = round(random.randrange(
                        0, self.DISPLAY_HEIGHT - self.SNAKE_BLOCK) / self.SNAKE_BLOCK) * self.SNAKE_BLOCK
                else:
                    self.CHECK_COORDINATES = False

        # MAX STEPS
        self.MAX_STEPS = 100

    # SHOWS SCORE
    def score_board(self, score):
        val = self.FONT_STYLE.render(
            "Snake score: " + str(score), True, self.BLACK)
        self.DISPLAY.blit(val, [10, 0])

    # SHOWS RECORD SCORE
    def record_score(self, record):
        val = self.FONT_STYLE.render(
            "Snake record score: " + str(record), True, self.BLACK)
        self.DISPLAY.blit(val, [10, 20])

    # SHOWS MAX STEPS
    def max_steps(self, steps):
        val = self.FONT_STYLE.render(
            str(steps) + " steps remaining", True, self.BLACK)
        self.DISPLAY.blit(val, [10, 40])

    # DRAW SNAKE
    def snake_segment(self, snake_block, snake_list):
        for x in snake_list:
            pygame.draw.rect(self.DISPLAY, self.BLUE, [
                x[0], x[1], snake_block, snake_block])
            pygame.draw.rect(self.DISPLAY, self.GREEN, [
                snake_list[self.LENGHT_OF_SNAKE - 1][0], snake_list[self.LENGHT_OF_SNAKE - 1][1], snake_block,
                snake_block])

    # GENERATE OBSERVATIONS
    def generate_observations(self):
        return self.GAME_CLOSE, self.LENGHT_OF_SNAKE - 1, [self.FOODX,
                                                           self.FOODY], self.SNAKE_LIST, self.LENGHT_OF_SNAKE

    # MAIN LOOP
    def game_loop(self, key):
        pygame.event.get()

        while self.GAME_CLOSE == True:
            self.DISPLAY.fill(self.WHITE)
            pygame.display.update()

        # MOVES
        if key == 0:  # LEFT
            self.SNAKEX_CHANGE = -self.SNAKE_BLOCK
            self.SNAKEY_CHANGE = 0

        if key == 1:  # RIGHT
            self.SNAKEX_CHANGE = self.SNAKE_BLOCK
            self.SNAKEY_CHANGE = 0

        if key == 2:  # UP
            self.SNAKEY_CHANGE = -self.SNAKE_BLOCK
            self.SNAKEX_CHANGE = 0

        if key == 3:  # DOWN
            self.SNAKEY_CHANGE = self.SNAKE_BLOCK
            self.SNAKEX_CHANGE = 0

        if self.SNAKEX >= self.DISPLAY_WIDHT or self.SNAKEX < 0 or self.SNAKEY >= self.DISPLAY_HEIGHT or self.SNAKEY < 0:
            self.GAME_CLOSE = True

        # CHANGE SNAKE HEAD COORDINATES
        self.SNAKEX += self.SNAKEX_CHANGE
        self.SNAKEY += self.SNAKEY_CHANGE

        # DISPLAY COLOR
        self.DISPLAY.fill(self.WHITE)

        # DRAW FOOD
        pygame.draw.rect(self.DISPLAY, self.RED, [
            self.FOODX, self.FOODY, self.SNAKE_BLOCK, self.SNAKE_BLOCK])

        # IF SNAKE TAKES FOOD
        if self.SNAKEX == self.FOODX and self.SNAKEY == self.FOODY:
            self.FOODX = round(random.randrange(
                0, self.DISPLAY_WIDHT - self.SNAKE_BLOCK) / self.SNAKE_BLOCK) * self.SNAKE_BLOCK
            self.FOODY = round(random.randrange(
                0, self.DISPLAY_HEIGHT - self.SNAKE_BLOCK) / self.SNAKE_BLOCK) * self.SNAKE_BLOCK

            self.LENGHT_OF_SNAKE += 1
            self.MAX_STEPS += 10

        # SNAKE HEAD
        SNAKE_HEAD = []
        SNAKE_HEAD.append(self.SNAKEX)
        SNAKE_HEAD.append(self.SNAKEY)
        self.SNAKE_LIST.append(SNAKE_HEAD)

        # DELETE SNAKE TAIL
        if len(self.SNAKE_LIST) > self.LENGHT_OF_SNAKE:
            del self.SNAKE_LIST[0]

        # IF SNAKE HIT ITSELF THE GAME ENDS
        for x in self.SNAKE_LIST[:-1]:
            if x == SNAKE_HEAD:
                self.GAME_CLOSE = True

        # DECREASES SNAKE STEPS
        self.MAX_STEPS -= 1

        self.snake_segment(self.SNAKE_BLOCK, self.SNAKE_LIST)
        self.score_board(self.LENGHT_OF_SNAKE - 1)
        self.record_score(self.RECORD)
        self.max_steps(self.MAX_STEPS)

        self.CHECK_COORDINATES = True

        # CHECKS IF THE APPLE ISNT GENERATE ON SNAKE POSITION
        while (self.CHECK_COORDINATES):
            for i in self.SNAKE_LIST:
                if i == [self.FOODX, self.FOODY]:
                    self.FOODX = round(random.randrange(
                        0, self.DISPLAY_WIDHT - self.SNAKE_BLOCK) / self.SNAKE_BLOCK) * self.SNAKE_BLOCK
                    self.FOODY = round(random.randrange(
                        0, self.DISPLAY_HEIGHT - self.SNAKE_BLOCK) / self.SNAKE_BLOCK) * self.SNAKE_BLOCK
                else:
                    self.CHECK_COORDINATES = False

        # UPDATE DISPLAY EVERY FRAME
        pygame.display.update()

        # FPS
        self.CLOCK.tick(self.SNAKE_SPEED)

        return self.generate_observations()


if __name__ == "__main__":
    game = SnakeGame()

    for _ in range(100):
        game.game_loop(random.randint(0, 3))
