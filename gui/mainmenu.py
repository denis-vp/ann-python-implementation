import pygame

from gui.button import Button
from gui.drawingboard import DrawingBoard

# Window settings
WINDOW_HEIGHT = 700
WINDOW_WIDTH = 900

# Button settings
BUTTON_HEIGHT = 50
BUTTON_WIDTH = WINDOW_WIDTH

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)


class MainMenu:
    def __init__(self):
        self.root = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Neural Network - Main Menu")

        self.open_drawing_board_button = Button(self.root, self.open_drawing_board, 0, 0, BUTTON_WIDTH, BUTTON_HEIGHT, PURPLE, "Open Drawing Board")
        self.buttons = [self.open_drawing_board_button]

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == pygame.BUTTON_LEFT:
                    for button in self.buttons:
                        button.click(pygame.mouse.get_pos())
            self.update_display()
        pygame.quit()
        quit()

    def update_display(self):
        self.root.fill(WHITE)
        for button in self.buttons:
            button.draw()
        pygame.display.update()

    def open_drawing_board(self):
        drawing_board = DrawingBoard()
        drawing_board.run()
