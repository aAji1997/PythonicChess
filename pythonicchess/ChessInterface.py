from ChessLogic import ConstrainedGameState
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
        self.width = self.height = 768
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
        """
        Converts pixel coordinates to board position and returns the position index.
        Args:
            x: The x-coordinate of the pixel.
            y: The y-coordinate of the pixel.
        Returns:
            The index of the board position corresponding to the pixel coordinates.
        """
        if self.white_to_move:
            row = y // self.square_size
            col = x // self.square_size
        else:
            row = (self.height - y) // self.square_size
            col = (self.width - x) // self.square_size
        position = row * 8 + col
        return position

    def load_images(self):
        """
        Load images for each piece on the board and store them in the 'images' dictionary.
        """
        pieces = list(self.board.keys())
        for piece in pieces:
            self.images[piece] = pg.transform.scale(pg.image.load(f'/home/hal/ssdlink/PythonicChess/pythonicchess/images/{piece}.png'), (self.square_size, self.square_size))
    
    def run(self):
        """
        The run function initializes the game, handles user input, and updates the game state. 
        It contains a game loop that checks for user input events, such as mouse clicks and quitting the game. 
        It also handles exceptions and prints any errors that occur during execution.
        """
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
                            if second_click_pos == self.first_click_pos:
                                self.first_click = False
                                self.first_click_pos = None
                                self.highlighted_square = None
                                continue
                            side = 'w' if self.white_to_move else 'b'
                            king_position = self.get_king_position(side)

                            if self.first_click_pos == king_position:
                                # Adjust column indices based on the side to move
                                kingside_col = 6 if self.white_to_move else 1
                                queenside_col = 2 if self.white_to_move else 5

                                if col == kingside_col:  # Kingside castling
                                    if side == 'w' and self.w_castle_k:
                                        self.move_piece(piece="OO")
                                    elif side == 'b' and self.b_castle_k:
                                        #print("Black kingside castling")
                                        self.move_piece(piece="OO")
                                    else:
                                        # Regular move
                                        self.move_piece(self.position_to_string(self.first_click_pos), self.position_to_string(second_click_pos))
                                elif col == queenside_col:  # Queenside castling
                                    if side == 'w' and self.w_castle_q:
                                        self.move_piece(piece="OOO")
                                    elif side == 'b' and self.b_castle_q:
                                        #print("Black queenside castling")
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
        """
        Reset the game state, display end game message if checkmated or drawn, sleep for 1 second,
        and then initialize the game with the start position.
        """
        self.draw_game_state()
        if self.checkmated:
            self.display_end_game_message('c')
        elif self.drawn:
            self.display_end_game_message('d')
        sleep(1)
        self.__init__()   
        self.set_start_position()
        
        
    def draw_board(self):
        """
        Draw the board on the screen using Pygame, with alternating colors for the squares.
        """
        colors = [pg.Color("white"), pg.Color(0, 100, 0)]
        for i in range(self.dimensions):
            for j in range(self.dimensions):
                if self.white_to_move:
                    color = colors[((i + j) % 2)]
                    pg.draw.rect(self.screen, color, (i * self.square_size, j * self.square_size, self.square_size, self.square_size))
                else:
                    color = colors[((i + j) % 2)]
                    pg.draw.rect(self.screen, color, ((7-i) * self.square_size, (7-j) * self.square_size, self.square_size, self.square_size))
    
    def draw_pieces(self):
        """
        Draws the pieces on the board based on the current state of the game. 
        This function iterates through each piece and its corresponding bitboard to determine the position of each piece on the screen. 
        If the white player is to move, the pieces are drawn normally. 
        If the black player is to move, the pieces are drawn with their positions flipped to reflect the black player's perspective. 
        """
        for piece, bitboard in self.board.items():
            for i in range(64):
                if (bitboard >> i) & 1:  # if the i-th bit is set
                    row = i // 8
                    col = i % 8
                    if self.white_to_move:
                        self.screen.blit(self.images[piece], pg.Rect(col * self.square_size, row * self.square_size, self.square_size, self.square_size))
                    else:
                        flipped_row = 7 - row
                        flipped_col = 7 - col
                        self.screen.blit(self.images[piece], pg.Rect(flipped_col * self.square_size, flipped_row * self.square_size, self.square_size, self.square_size))
                        
    def highlight_square(self, square):
        """
        Highlights a square on the screen.

        Parameters:
            self (object): The object instance
            square (tuple): The coordinates of the square to highlight

        Returns:
            None
        """
        row, col = square
        pg.draw.rect(self.screen, (255, 0, 0), (col * self.square_size, row * self.square_size, self.square_size, self.square_size), 3)
    
    def display_end_game_message(self, end_state):
        """
        Displays an end game message ("Draw" or "Checkmate") on the Pygame window.
        
        :param end_state: A string indicating the end state of the game ('c' for checkmate, 'd' for draw).
        """
        font = pg.font.SysFont(name='Arial', size=48)  # You can choose a different font and size
        if end_state == 'c':
            message = 'Checkmate'
        elif end_state == 'd':
            message = 'Draw'
        else:
            return  # Invalid end_state, do nothing
        
        text = font.render(message, True, pg.Color('Red'))  # Render the text in red
        text_rect = text.get_rect(center=(self.width // 2, self.height // 2))  # Center the text
        
        self.screen.blit(text, text_rect)  # Draw the text on the screen
        pg.display.update()
    
    def render_pawn_promotion(self):
        """
        Render the pawn promotion window and handle piece selection events.
        """
        if not self.promoting:
            return

        # Define the promotion pane dimensions
        pane_width = pane_height = 768
        

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