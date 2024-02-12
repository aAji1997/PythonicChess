from ChessEngine import ConstrainedGameState
import numpy as np
import pygame as pg
import sys
import signal
from time import sleep

def signal_handler(sig, frame):
    print('You pressed Ctrl+C or another exit command!')
    sys.exit(0)
    
"""
This is the main driver file for the chess game. 
It will be responsible for handling user input and displaying the current game state.
"""
class GameInterface(ConstrainedGameState):
    """
    This class is used to handle user input and display the current game state.
    """
    def __init__(self):
        super().__init__()
        self.set_start_position()
        self.width = self.height = 512
        self.dimensions = 8
        self.square_size = self.height // self.dimensions
        self.fps = 15
        self.images = {}
        self.screen = pg.display.set_mode((self.width, self.height))
        self.load_images()
        self.first_click = False
        self.first_click_pos = None
        self.reset = False
        
    def pixel_to_position(self, x, y):
        row = y // self.square_size
        col = x // self.square_size
        position = row * 8 + col
        return position

    def load_images(self):
        pieces = list(self.board.keys())
        for piece in pieces:
            self.images[piece] = pg.transform.scale(pg.image.load(f'/home/hal/ssdlink/PythonicChess/pythonicchess/images/{piece}.png'), (self.square_size, self.square_size))
    
    def run(self):
        pg.init()
        clock = pg.time.Clock()
        running = True
        self.highlighted_square = None
        try:
            while running:
                if self.reset:
                    self.reset_game()
                self.draw_game_state()
                
                if self.checkmated or self.drawn:
                    self.reset = True
                    self.draw_game_state()
                    
                if self.highlighted_square:
                    self.highlight_square(self.highlighted_square)
                for e in pg.event.get():
                    if e.type == pg.QUIT:
                        running = False
                        pg.quit()
                        sys.exit()
                    elif e.type == pg.MOUSEBUTTONDOWN:
                        mouse_x, mouse_y = pg.mouse.get_pos()
                        col = mouse_x // self.square_size
                        row = mouse_y // self.square_size
                        if not self.first_click:
                            self.first_click_pos = self.pixel_to_position(mouse_x, mouse_y)
                            self.first_click = True
                            self.highlighted_square = (row, col)
                        else:
                            second_click_pos = self.pixel_to_position(mouse_x, mouse_y)
                            side = 'w' if self.white_to_move else 'b'
                            king_position = self.get_king_position(side)
                            if self.first_click_pos == king_position:
                                # Check if the move is a kingside or queenside castling
                                if col == 6:  # Kingside castling
                                    if side == 'w' and self.w_castle_k:
                                        self.move_piece(piece="OO")
                                    elif side == 'b' and self.b_castle_k:
                                        self.move_piece(piece="OO")
                                    else:
                                        # Regular move
                                        self.move_piece(self.position_to_string(self.first_click_pos), self.position_to_string(second_click_pos))
                                        
                                elif col == 2:  # Queenside castling
                                    if side == 'w' and self.w_castle_q:
                                        self.move_piece(piece="OOO")
                                    elif side == 'b' and self.b_castle_q:
                                        self.move_piece(piece="OOO")
                                    else:
                                        # Regular move
                                        self.move_piece(self.position_to_string(self.first_click_pos), self.position_to_string(second_click_pos))
                                else:
                                    # Regular move
                                    self.move_piece(self.position_to_string(self.first_click_pos), self.position_to_string(second_click_pos))
                            else:
                                # Regular move
                                self.move_piece(self.position_to_string(self.first_click_pos), self.position_to_string(second_click_pos))
                            self.first_click = False
                            self.highlighted_square = None
                self.render_pawn_promotion()         
                clock.tick(self.fps)
                pg.display.update()
        except Exception as e:
            print(e)

                
    def draw_game_state(self):
        self.draw_board()
        self.draw_pieces()

    def reset_game(self):
        self.draw_game_state()
        sleep(1)   
        self.set_start_position()
        self.reset = False
        self.checkmated = False
        
    def draw_board(self):
        colors = [pg.Color("white"), pg.Color(0, 100, 0)]
        for i in range(self.dimensions):
            for j in range(self.dimensions):
                color = colors[((i + j) % 2)]
                pg.draw.rect(self.screen, color, (i * self.square_size, j * self.square_size, self.square_size, self.square_size))
    
    def draw_pieces(self):
        for piece, bitboard in self.board.items():
            for i in range(64):
                if (bitboard >> i) & 1:  # if the i-th bit is set
                    row = i // 8
                    col = i % 8
                    self.screen.blit(self.images[piece], pg.Rect(col * self.square_size, row * self.square_size, self.square_size, self.square_size))
    def highlight_square(self, square):
        row, col = square
        pg.draw.rect(self.screen, (255, 0, 0), (col * self.square_size, row * self.square_size, self.square_size, self.square_size), 3)
    
    def render_pawn_promotion(self):
        if not self.promoting:
            return

        # Define the promotion pane dimensions
        pane_width = 512
        pane_height = 512

        # Create a new surface for the promotion pane
        promotion_surface = pg.Surface((pane_width, pane_height))
        promotion_surface.fill(pg.Color(200, 200, 200))

        # Load and display the promotion piece options
        pieces = ['Q', 'R', 'B', 'N']
        piece_images = {}
        for piece in pieces:
            color_prefix = 'w' if not self.white_to_move else 'b'
            piece_string = f'{color_prefix}{piece}'
            piece_images[piece_string] = pg.transform.scale(
                pg.image.load(f'/home/hal/ssdlink/PythonicChess/pythonicchess/images/{piece_string}.png'),
                (self.square_size, self.square_size)
            )
            piece_x = pieces.index(piece) * (pane_width // len(pieces))
            promotion_surface.blit(piece_images[piece_string], (piece_x, 0))

        # Create a new window for the promotion pane
        promotion_window = pg.display.set_mode((pane_width, pane_height))
        promotion_window.blit(promotion_surface, (0, 0))
        pg.display.update()
    # Event handling for piece selection
        while True:
            for e in pg.event.get():
                if e.type == pg.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pg.mouse.get_pos()
                    for i, piece in enumerate(pieces):
                        piece_x = i * (pane_width // len(pieces))
                        if piece_x < mouse_x < piece_x + (pane_width // len(pieces)):
                            self.promoted_piece = f"{color_prefix}{piece}"
                            color = "b" if self.white_to_move else "w"
                            self.clear_piece(piece=color+"P", position=self.promoting_from_pos)
                            self.set_piece(piece=self.promoted_piece, position=self.promoting_to_pos)
                            move_array = np.array([self.piece_enum["="+self.promoted_piece], self.promoting_from_pos, self.promoting_to_pos], dtype=np.int8)
                            self.move_log = np.vstack([self.move_log, move_array])
                            self.promoting = False
                            
                            return
                        
                elif e.type == pg.QUIT:
                    pg.display.quit()
                    return
        
if __name__ == "__main__":
    game = GameInterface()
    signal.signal(signal.SIGINT, signal_handler)
    game.run()