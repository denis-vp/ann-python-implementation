import pygame

from gui.cell import Cell
from settings import WINDOW_HEIGHT, WINDOW_WIDTH


# The grid is a square
GRID_HEIGHT = WINDOW_HEIGHT
GRID_WIDTH = GRID_HEIGHT

NUM_ROWS, NUM_COLUMNS = 28, 28
CELL_HEIGHT = GRID_HEIGHT // NUM_ROWS
CELL_WIDTH = CELL_HEIGHT


class GUI:
    def __init__(self):
        self.root = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Neural Network - Draw")

        self.cells: list[Cell] = []
        self.create_grid(NUM_ROWS, NUM_COLUMNS)

    def create_grid(self, num_rows: int, num_columns: int):
        for row in range(num_rows):
            for col in range(num_columns):
                self.cells.append(Cell(row * CELL_WIDTH, col * CELL_HEIGHT, 255))
        return self.cells

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == pygame.BUTTON_LEFT:
                    self.draw(pygame.mouse.get_pos())
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == pygame.BUTTON_RIGHT:
                    self.erase(pygame.mouse.get_pos())
                if event.type == pygame.MOUSEMOTION and event.buttons[0]:
                    self.draw(pygame.mouse.get_pos())
                if event.type == pygame.MOUSEMOTION and event.buttons[2]:
                    self.erase(pygame.mouse.get_pos())
            self.update_display()
        pygame.quit()
        quit()

    #       Display Updating

    def update_display(self):
        self.draw_grid()
        pygame.display.update()

    def draw_grid(self):
        for cell in self.cells:
            cell_color = (cell.value, cell.value, cell.value)
            pygame.draw.rect(self.root, cell_color, (cell.x, cell.y, 50, 50))

    #       Interaction

    def draw(self, coordinates: tuple[int, int]):
        x, y = coordinates
        for cell in self.cells:
            if cell.x <= x <= cell.x + CELL_WIDTH and cell.y <= y <= cell.y + CELL_HEIGHT:
                cell.value = 0
                break

    def erase(self, coordinates: tuple[int, int]):
        x, y = coordinates
        for cell in self.cells:
            if cell.x <= x <= cell.x + CELL_WIDTH and cell.y <= y <= cell.y + CELL_HEIGHT:
                cell.value = 0
                break
