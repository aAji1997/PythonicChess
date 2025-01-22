from ChessLogic import ConstrainedGameState
import numpy as np
import pygame as pg
import sys
import signal
from time import sleep
from ChessEngine import GameEngine
from copy import deepcopy
import traceback

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
        self.highlighted_moves = set()  # Store positions of valid moves to highlight
        self.game_mode = None  # 'human' or 'computer'
        self.engine = GameEngine()  # Initialize the chess engine
        self.computer_thinking = False
        self.player_color = 'w'  # Default to white, will be set in color selection
        self.board_orientation = 'w'  # Board orientation, white at bottom by default
        self.highlighted_square = None  # Initialize highlighted square attribute
        self.last_computer_move = None  # Track computer's last move (from_pos, to_pos)
        # Time control attributes
        self.time_control = None  # 'blitz', 'rapid', or 'long'
        self.white_time = 0  # Time in milliseconds
        self.black_time = 0  # Time in milliseconds
        self.increment = 0  # Time increment in milliseconds
        self.last_move_time = None  # Time of the last move
        self.first_move_made = False  # Track if first move has been made
        # PGN move tracking
        self.pgn_moves = []  # List to store moves in PGN format
        self.move_window = None  # Separate window for displaying moves
        self.move_window_width = 300
        self.move_window_height = 768
        self.move_font = None
        self.move_surface = None
        self.scroll_y = 0  # For scrolling through moves
        self.max_scroll = 0
        
    def pixel_to_position(self, x, y):
        """
        Converts pixel coordinates to board position and returns the position index.
        Args:
            x: The x-coordinate of the pixel.
            y: The y-coordinate of the pixel.
        Returns:
            The index of the board position corresponding to the pixel coordinates.
        """
        row = y // self.square_size
        col = x // self.square_size
        if self.board_orientation == 'b':  # Only flip if board is oriented from black's perspective
            row = 7 - row
            col = 7 - col
        position = row * 8 + col
        return position

    def load_images(self):
        """
        Load images for each piece on the board and store them in the 'images' dictionary.
        """
        pieces = list(self.board.keys())
        for piece in pieces:
            self.images[piece] = pg.transform.scale(pg.image.load(f'/home/hal/ssdlink/projs/PythonicChess/pythonicchess/images/{piece}.png'), (self.square_size, self.square_size))

    def show_game_mode_selection(self):
        """
        Display the game mode selection screen and return the selected mode.
        """
        # Colors
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GREEN = (0, 100, 0)
        HOVER_GREEN = (0, 120, 0)
        
        # Create buttons
        button_width = 300
        button_height = 80
        button_x = self.width // 2 - button_width // 2
        human_button_y = self.height // 2 - button_height - 20
        computer_button_y = self.height // 2 + 20
        
        # Font
        pg.font.init()
        font = pg.font.SysFont('Arial', 32)
        title_font = pg.font.SysFont('Arial', 48)
        
        # Title text
        title = title_font.render('Select Game Mode', True, BLACK)
        title_rect = title.get_rect(center=(self.width // 2, self.height // 3))
        
        while True:
            mouse_pos = pg.mouse.get_pos()
            
            # Check if mouse is hovering over buttons
            human_hover = button_x <= mouse_pos[0] <= button_x + button_width and human_button_y <= mouse_pos[1] <= human_button_y + button_height
            computer_hover = button_x <= mouse_pos[0] <= button_x + button_width and computer_button_y <= mouse_pos[1] <= computer_button_y + button_height
            
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
                if event.type == pg.MOUSEBUTTONDOWN:
                    if human_hover:
                        return 'human'
                    elif computer_hover:
                        return 'computer'
            
            # Draw screen
            self.screen.fill(WHITE)
            
            # Draw title
            self.screen.blit(title, title_rect)
            
            # Draw buttons
            pg.draw.rect(self.screen, HOVER_GREEN if human_hover else GREEN, (button_x, human_button_y, button_width, button_height))
            pg.draw.rect(self.screen, HOVER_GREEN if computer_hover else GREEN, (button_x, computer_button_y, button_width, button_height))
            
            # Draw button text
            human_text = font.render('Human vs Human', True, WHITE)
            computer_text = font.render('Human vs Computer', True, WHITE)
            
            human_text_rect = human_text.get_rect(center=(self.width // 2, human_button_y + button_height // 2))
            computer_text_rect = computer_text.get_rect(center=(self.width // 2, computer_button_y + button_height // 2))
            
            self.screen.blit(human_text, human_text_rect)
            self.screen.blit(computer_text, computer_text_rect)
            
            pg.display.update()

    def show_color_selection(self):
        """
        Display the color selection screen and return the selected color.
        """
        # Colors
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GREEN = (0, 100, 0)
        HOVER_GREEN = (0, 120, 0)
        
        # Create buttons
        button_width = 300
        button_height = 80
        button_x = self.width // 2 - button_width // 2
        white_button_y = self.height // 2 - button_height - 20
        black_button_y = self.height // 2 + 20
        
        # Font
        pg.font.init()
        font = pg.font.SysFont('Arial', 32)
        title_font = pg.font.SysFont('Arial', 48)
        
        # Title text
        title = title_font.render('Select Your Color', True, BLACK)
        title_rect = title.get_rect(center=(self.width // 2, self.height // 3))
        
        while True:
            mouse_pos = pg.mouse.get_pos()
            
            # Check if mouse is hovering over buttons
            white_hover = button_x <= mouse_pos[0] <= button_x + button_width and white_button_y <= mouse_pos[1] <= white_button_y + button_height
            black_hover = button_x <= mouse_pos[0] <= button_x + button_width and black_button_y <= mouse_pos[1] <= black_button_y + button_height
            
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
                if event.type == pg.MOUSEBUTTONDOWN:
                    if white_hover:
                        return 'w'
                    elif black_hover:
                        return 'b'
            
            # Draw screen
            self.screen.fill(WHITE)
            
            # Draw title
            self.screen.blit(title, title_rect)
            
            # Draw buttons
            pg.draw.rect(self.screen, HOVER_GREEN if white_hover else GREEN, (button_x, white_button_y, button_width, button_height))
            pg.draw.rect(self.screen, HOVER_GREEN if black_hover else GREEN, (button_x, black_button_y, button_width, button_height))
            
            # Draw button text
            white_text = font.render('Play as White', True, WHITE)
            black_text = font.render('Play as Black', True, WHITE)
            
            white_text_rect = white_text.get_rect(center=(self.width // 2, white_button_y + button_height // 2))
            black_text_rect = black_text.get_rect(center=(self.width // 2, black_button_y + button_height // 2))
            
            self.screen.blit(white_text, white_text_rect)
            self.screen.blit(black_text, black_text_rect)
            
            pg.display.update()

    def make_computer_move(self):
        """
        Calculate and make the computer's move using the chess engine.
        """
        computer_color = 'b' if self.player_color == 'w' else 'w'
        if self.white_to_move and computer_color == 'w' or not self.white_to_move and computer_color == 'b':
            if not (self.checkmated or self.drawn):
                self.computer_thinking = True
                print("\n=== Debug: Starting computer move calculation ===")
                
                # Start tracking computer's thinking time
                computer_start_time = pg.time.get_ticks()
                last_display_update = computer_start_time
                
                # Calculate time to use for this move
                if self.time_control and self.first_move_made:
                    remaining_time = self.white_time if self.white_to_move else self.black_time
                    increment = self.increment
                    moves_left = max(40 - len(self.move_log), 10)  # Estimate remaining moves, minimum 10
                    
                    # Base time per move: remaining time divided by estimated moves left
                    base_time = remaining_time / moves_left
                    
                    # Add a portion of the increment to the base time
                    base_time += increment * 0.8  # Use 80% of the increment
                    
                    # Adjust based on game phase and remaining time
                    if len(self.move_log) < 10:  # Opening
                        max_time = min(base_time * 0.5, 5000)  # Max 5 seconds in opening
                    elif remaining_time < 10000:  # Less than 10 seconds left
                        max_time = min(remaining_time * 0.1, 1000)  # Use at most 10% of remaining time
                    else:
                        max_time = min(base_time * 1.5, remaining_time * 0.2)  # Use at most 20% of remaining time
                    
                    # Ensure minimum and maximum thinking time
                    max_time = max(min(max_time, remaining_time * 0.2), 100)  # Between 0.1 and 20% of remaining time
                else:
                    max_time = 5000  # Default 5 seconds if no time control
                
                print(f"\n=== Debug: Allocated thinking time: {max_time/1000:.2f} seconds ===")
                
                # Sync all necessary game state with the engine
                self.engine.board = deepcopy(self.board)
                self.engine.white_to_move = self.white_to_move
                self.engine.w_castle_k = self.w_castle_k
                self.engine.w_castle_q = self.w_castle_q
                self.engine.b_castle_k = self.b_castle_k
                self.engine.b_castle_q = self.b_castle_q
                self.engine.move_log = self.move_log.copy()
                self.engine.board_states = self.board_states.copy()
                
                # Get the best move using iterative deepening
                try:
                    # Create a clock for frame timing
                    clock = pg.time.Clock()
                    
                    def update_display():
                        # Draw the current state
                        self.draw_game_state()
                        self.display_thinking_message()
                        pg.display.update()
                        # Process events to keep the window responsive
                        for event in pg.event.get():
                            if event.type == pg.QUIT:
                                pg.quit()
                                sys.exit()
                    
                    # Initial display update
                    update_display()
                    
                    # Start the search with periodic display updates
                    search_started = pg.time.get_ticks()
                    best_move = None
                    time_slice = min(100, max_time * 0.1)  # Use smaller time slices for shorter max_time
                    
                    while (pg.time.get_ticks() - search_started) < max_time:
                        current_best = self.engine.iterative_deepening_search(self.engine.board, max_time=time_slice/1000)  # Convert to seconds
                        if current_best:
                            best_move = current_best
                        
                        # Update display every 100ms
                        current_time = pg.time.get_ticks()
                        if current_time - last_display_update >= 100:
                            update_display()
                            last_display_update = current_time
                            clock.tick(30)  # Limit to 30 FPS
                    
                    print("\n=== Debug: Search completed successfully ===")
                except Exception as e:
                    print(f"\n=== Debug: Error in search: {str(e)} ===")
                    print("Stack trace:")
                    import traceback
                    traceback.print_exc()
                    self.computer_thinking = False
                    return
                
                # Calculate final thinking time and update clock
                if self.time_control and self.first_move_made:
                    computer_end_time = pg.time.get_ticks()
                    thinking_time = computer_end_time - computer_start_time
                    if self.white_to_move:
                        self.white_time = max(0, self.white_time - thinking_time)
                    else:
                        self.black_time = max(0, self.black_time - thinking_time)
                
                if best_move:
                    print(f"\n=== Debug: Best move found: {best_move} ===")
                    piece, from_pos, to_pos, promotion_piece = best_move
                    # Store the move for highlighting
                    self.last_computer_move = (from_pos, to_pos)
                    if promotion_piece:
                        self.promoted_piece = computer_color + promotion_piece
                        self.promoting_from_pos = from_pos
                        self.promoting_to_pos = to_pos
                        self.promoting = True
                        self.move_piece(self.position_to_string(from_pos), self.position_to_string(to_pos))
                    else:
                        self.move_piece(self.position_to_string(from_pos), self.position_to_string(to_pos))
                else:
                    print("\n=== Debug: No valid move found ===")
                
                self.computer_thinking = False

    def display_thinking_message(self):
        """
        Display a "Computer is thinking..." message on the screen with a semi-transparent background.
        """
        font = pg.font.SysFont('Arial', 36)
        text = font.render('Computer is thinking...', True, pg.Color('Red'))
        text_rect = text.get_rect(center=(self.width // 2, 30))
        
        # Create background box
        padding = 10  # Smaller padding for the thinking message
        box_rect = pg.Rect(text_rect.left - padding,
                          text_rect.top - padding,
                          text_rect.width + 2 * padding,
                          text_rect.height + 2 * padding)
        
        # Create a semi-transparent surface for the background
        background_surface = pg.Surface((box_rect.width, box_rect.height), pg.SRCALPHA)
        background_color = (255, 255, 255, 180)  # White with 180/255 alpha (semi-transparent)
        pg.draw.rect(background_surface, background_color, background_surface.get_rect())
        
        # Draw the semi-transparent background and border
        self.screen.blit(background_surface, box_rect)
        pg.draw.rect(self.screen, pg.Color('Black'), box_rect, 2)  # 2-pixel border
        
        # Draw the text
        self.screen.blit(text, text_rect)
    
    def run(self):
        """
        The run function initializes the game, handles user input, and updates the game state.
        """
        pg.init()
        clock = pg.time.Clock()
        
        # Show game mode selection screen
        self.game_mode = self.show_game_mode_selection()
        
        # Show time control selection screen
        self.time_control = self.show_time_control_selection()
        # Don't start the clock until first move is made
        self.last_move_time = None
        
        # If computer mode, show color selection
        if self.game_mode == 'computer':
            self.player_color = self.show_color_selection()
            self.board_orientation = self.player_color  # Set board orientation to match player color
            
            # If player chose black, make first move as white
            if self.player_color == 'b':
                self.make_computer_move()
        
        running = True
        self.highlighted_square = None
        try:
            while running:
                if self.reset:
                    self.reset_game()
                    # Show game mode selection again after reset
                    self.game_mode = self.show_game_mode_selection()
                    # Show time control selection again after reset
                    self.time_control = self.show_time_control_selection()
                    self.last_move_time = None  # Don't start clock until first move
                    self.first_move_made = False  # Reset first move flag
                    
                    if self.game_mode == 'computer':
                        self.player_color = self.show_color_selection()
                        self.board_orientation = self.player_color
                        if self.player_color == 'b':
                            self.make_computer_move()
                
                # Check for time out before making computer move
                if self.first_move_made and self.check_time_out():
                    sleep(2)  # Give players time to see the message
                    self.reset = True
                    continue
                
                # Make computer move if it's computer's turn
                if self.game_mode == 'computer':
                    self.make_computer_move()
                
                self.draw_game_state()
                
                if self.checkmated or self.drawn:
                    if self.checkmated:
                        self.display_end_game_message('c')
                    else:
                        self.display_end_game_message('d')
                    pg.display.update()
                    sleep(2)  # Give players time to see the message
                    self.reset = True
                    self.draw_game_state()
                    
                if self.highlighted_square:
                    self.highlight_square(self.highlighted_square)
                    
                for e in pg.event.get():
                    if e.type == pg.QUIT:
                        running = False
                        pg.quit()
                        sys.exit()
                    elif e.type == pg.MOUSEWHEEL:
                        self.handle_scroll(e)
                    elif e.type == pg.MOUSEBUTTONDOWN and (self.game_mode == 'human' or 
                            (self.game_mode == 'computer' and 
                             (self.white_to_move and self.player_color == 'w' or 
                              not self.white_to_move and self.player_color == 'b'))):
                        try:
                            mouse_x, mouse_y = pg.mouse.get_pos()
                            # Only process board clicks if mouse is in the board area
                            if mouse_x < self.width:
                                col = mouse_x // self.square_size
                                row = mouse_y // self.square_size

                                if not self.first_click:
                                    self.first_click_pos = self.pixel_to_position(mouse_x, mouse_y)
                                    try:
                                        self.highlighted_moves = self.get_valid_moves(self.first_click_pos)
                                    except Exception as e:
                                        print(f"Error getting valid moves: {str(e)}")
                                        import traceback
                                        traceback.print_exc()
                                        continue
                                    self.first_click = True
                                    self.highlighted_square = (row, col)
                                else:
                                    second_click_pos = self.pixel_to_position(mouse_x, mouse_y)
                                    if second_click_pos == self.first_click_pos:
                                        self.first_click = False
                                        self.first_click_pos = None
                                        self.highlighted_square = None
                                        self.highlighted_moves.clear()
                                        continue
                                    side = 'w' if self.white_to_move else 'b'
                                    king_position = self.get_king_position(side)

                                    if self.first_click_pos == king_position:
                                        kingside_col = 6 if self.white_to_move else 1
                                        queenside_col = 2 if self.white_to_move else 5

                                        if col == kingside_col:
                                            if side == 'w' and self.w_castle_k:
                                                self.move_piece(piece="OO")
                                            elif side == 'b' and self.b_castle_k:
                                                self.move_piece(piece="OO")
                                            else:
                                                self.move_piece(self.position_to_string(self.first_click_pos), self.position_to_string(second_click_pos))
                                        elif col == queenside_col:
                                            if side == 'w' and self.w_castle_q:
                                                self.move_piece(piece="OOO")
                                            elif side == 'b' and self.b_castle_q:
                                                self.move_piece(piece="OOO")
                                            else:
                                                self.move_piece(self.position_to_string(self.first_click_pos), self.position_to_string(second_click_pos))
                                        else:
                                            self.move_piece(self.position_to_string(self.first_click_pos), self.position_to_string(second_click_pos))

                                    else:
                                        self.move_piece(self.position_to_string(self.first_click_pos), self.position_to_string(second_click_pos))

                                    self.first_click = False
                                    self.highlighted_square = None
                                    self.highlighted_moves.clear()
                        except Exception as e:
                            print(f"Error processing mouse click: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            
                self.render_pawn_promotion()
                clock.tick(self.fps)
                pg.display.update()
        except Exception as e:
            print(f"Error during game loop: {str(e)}")
            import traceback
            traceback.print_exc()

                
    def draw_game_state(self):
        """
        Draw the complete game state in the correct order:
        1. Board squares
        2. Move highlights
        3. Pieces
        4. Square selection highlight
        5. Last computer move highlight
        6. Clock display
        """
        self.draw_board()  # Draw the base board first
        self.draw_highlighted_moves()  # Draw move highlights
        self.draw_pieces()  # Draw pieces on top
        if self.highlighted_square:  # Draw selection highlight last
            self.highlight_square(self.highlighted_square)
        if self.last_computer_move and self.game_mode == 'computer':  # Draw computer's last move highlight
            self.highlight_last_computer_move()
        if self.time_control:  # Draw clock if time control is active
            self.draw_clock()
            
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
        self.last_computer_move = None  # Clear the last computer move
        self.pgn_moves = []  # Clear PGN moves
        self.scroll_y = 0  # Reset scroll position
        
        
    def draw_board(self):
        """
        Draw the board on the screen using Pygame, with alternating colors for the squares.
        """
        colors = [pg.Color("white"), pg.Color(0, 100, 0)]
        for i in range(self.dimensions):
            for j in range(self.dimensions):
                if self.board_orientation == 'w':
                    color = colors[((i + j) % 2)]
                    pg.draw.rect(self.screen, color, (i * self.square_size, j * self.square_size, self.square_size, self.square_size))
                else:
                    color = colors[((i + j) % 2)]
                    pg.draw.rect(self.screen, color, ((7-i) * self.square_size, (7-j) * self.square_size, self.square_size, self.square_size))
    
    def draw_pieces(self):
        """
        Draws the pieces on the board based on the current state of the game. 
        """
        for piece, bitboard in self.board.items():
            for i in range(64):
                if (bitboard >> i) & 1:  # if the i-th bit is set
                    row = i // 8
                    col = i % 8
                    if self.board_orientation == 'b':  # Only flip if board is oriented from black's perspective
                        row = 7 - row
                        col = 7 - col
                    self.screen.blit(self.images[piece], pg.Rect(col * self.square_size, row * self.square_size, self.square_size, self.square_size))
                        
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
    
    def display_end_game_message(self, end_state, winner=None):
        """
        Displays an end game message ("Draw", "Checkmate", or "Time Out") on the Pygame window.
        
        :param end_state: A string indicating the end state of the game ('c' for checkmate, 'd' for draw, 't' for time out).
        :param winner: The winning side ('w' or 'b') for time out situations.
        """
        font = pg.font.SysFont(name='Arial', size=48)
        if end_state == 'c':
            # In checkmate, the side that just moved won
            winner = "Black" if self.white_to_move else "White"
            message = f'Checkmate - {winner} wins!'
        elif end_state == 'd':
            message = 'Draw'
        elif end_state == 't':
            winner = "White" if winner == 'w' else "Black"
            message = f'Time Out - {winner} wins!'
        else:
            return  # Invalid end_state, do nothing
        
        # Create text surface
        text = font.render(message, True, pg.Color('Red'))
        text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
        
        # Create background box
        padding = 20
        box_rect = pg.Rect(text_rect.left - padding,
                          text_rect.top - padding,
                          text_rect.width + 2 * padding,
                          text_rect.height + 2 * padding)
        
        # Draw white background box with black border
        pg.draw.rect(self.screen, pg.Color('White'), box_rect)
        pg.draw.rect(self.screen, pg.Color('Black'), box_rect, 2)
        
        # Draw the text
        self.screen.blit(text, text_rect)
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
                pg.image.load(f'/home/hal/ssdlink/projs/PythonicChess/pythonicchess/images/{piece_string}.png'),
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
        
    def get_valid_moves(self, position):
        """
        Get all valid moves for the piece at the given position.
        Returns a set of valid destination positions.
        """
        valid_moves = set()
        piece = None
        side = 'w' if self.white_to_move else 'b'
        
        # Find which piece is at this position
        for piece_name, bitboard in self.board.items():
            if (bitboard >> position) & 1:
                if piece_name[0] == side:  # Only get moves for pieces of the current side
                    piece = piece_name
                break
                
        if not piece:
            return valid_moves
            
        # For pawns, we need to check one or two squares ahead and diagonals for captures
        if piece[1] == 'P':
            direction = -8 if side == 'w' else 8
            start_rank = 6 if side == 'w' else 1
            
            # One square ahead
            one_ahead = position + direction
            if 0 <= one_ahead < 64 and not self.get_piece_at_position(one_ahead):
                # Check if move is valid without actually making it
                from_square = self.position_to_string(position)
                to_square = self.position_to_string(one_ahead)
                try:
                    # Save board state
                    old_board = self.board.copy()
                    if self.piece_constrainer(from_square=from_square, to_square=to_square, piece=piece):
                        # Test if this move would leave us in check
                        self.clear_piece(piece=piece, position=position)
                        self.set_piece(piece=piece, position=one_ahead)
                        if not self.determine_if_checked(side=side):
                            valid_moves.add(one_ahead)
                    # Restore board state
                    self.board = old_board
                except:
                    # Restore board state
                    self.board = old_board
                    
                # Two squares ahead from starting position
                if position // 8 == start_rank:
                    two_ahead = one_ahead + direction
                    if 0 <= two_ahead < 64 and not self.get_piece_at_position(two_ahead):
                        to_square = self.position_to_string(two_ahead)
                        try:
                            # Save board state
                            old_board = self.board.copy()
                            if self.piece_constrainer(from_square=from_square, to_square=to_square, piece=piece):
                                # Test if this move would leave us in check
                                self.clear_piece(piece=piece, position=position)
                                self.set_piece(piece=piece, position=two_ahead)
                                if not self.determine_if_checked(side=side):
                                    valid_moves.add(two_ahead)
                            # Restore board state
                            self.board = old_board
                        except:
                            # Restore board state
                            self.board = old_board
                
            # Diagonal captures and en passant
            file = position % 8
            captures = []
            if file > 0:  # Can capture to the left
                captures.append(position + direction - 1)
            if file < 7:  # Can capture to the right
                captures.append(position + direction + 1)
                
            for capture_pos in captures:
                if 0 <= capture_pos < 64:
                    capture_piece = self.get_piece_at_position(capture_pos)
                    from_square = self.position_to_string(position)
                    to_square = self.position_to_string(capture_pos)
                    
                    # Regular capture
                    if capture_piece and capture_piece[0] != side:
                        try:
                            # Save board state
                            old_board = self.board.copy()
                            if self.piece_constrainer(from_square=from_square, to_square=to_square, piece=piece):
                                # Test if this move would leave us in check
                                captured_piece = self.get_piece_at_position(capture_pos)
                                self.clear_piece(piece=piece, position=position)
                                if captured_piece:
                                    self.clear_piece(piece=captured_piece, position=capture_pos)
                                self.set_piece(piece=piece, position=capture_pos)
                                if not self.determine_if_checked(side=side):
                                    valid_moves.add(capture_pos)
                            # Restore board state
                            self.board = old_board
                        except:
                            # Restore board state
                            self.board = old_board
                
            # Check for en passant separately from regular captures
            # Get the last move from the move log
            last_move = self.move_log[-1] if self.move_log.size > 0 and self.move_log[-1][0] != -1 else None
            if last_move is not None:
                try:
                    last_piece = list(self.piece_enum.keys())[list(self.piece_enum.values()).index(last_move[0])]
                    last_from_position = last_move[1]
                    last_to_position = last_move[2]
                    
                    # Check if last move was a two-square pawn advance
                    if last_piece[1].lower() == 'p' and abs(last_from_position - last_to_position) == 16:
                        # For en passant, we need to check if our pawn is adjacent to the pawn that just moved
                        last_pawn_file = last_to_position % 8
                        our_pawn_file = position % 8
                        
                        # Check if our pawn is on an adjacent file and the correct rank
                        if abs(last_pawn_file - our_pawn_file) == 1 and ((side == 'b' and position // 8 == last_to_position // 8) or 
                                                                         (side == 'w' and position // 8 == last_to_position // 8)):
                            # Calculate en passant target square (the square behind the pawn that moved two squares)
                            ep_target = last_to_position + (8 if side == 'b' else -8)
                            
                            # The capture position should be the square behind the pawn that moved
                            for capture_pos in captures:
                                if capture_pos == ep_target:
                                    try:
                                        # Save board state
                                        old_board = self.board.copy()
                                        # Test if this move would leave us in check
                                        captured_pawn_pos = last_to_position
                                        captured_pawn = self.get_piece_at_position(captured_pawn_pos)
                                        self.clear_piece(piece=piece, position=position)
                                        self.clear_piece(piece=captured_pawn, position=captured_pawn_pos)
                                        self.set_piece(piece=piece, position=capture_pos)
                                        if not self.determine_if_checked(side=side):
                                            valid_moves.add(capture_pos)
                                        # Restore board state
                                        self.board = old_board
                                    except Exception as e:
                                        # Restore board state
                                        self.board = old_board
                except (ValueError, IndexError) as e:
                    pass
                
            return valid_moves
            
        # Get the lookup table based on piece type
        elif piece[1] == 'N':
            lookup_table = self.N_lookup
        elif piece[1] == 'B':
            lookup_table = self.B_lookup
        elif piece[1] == 'R':
            lookup_table = self.R_lookup
        elif piece[1] == 'Q':
            lookup_table = self.Q_lookup
        elif piece[1] == 'K':
            # Special handling for king
            for to_pos in range(64):
                try:
                    # Skip if destination has piece of same color
                    target_piece = self.get_piece_at_position(to_pos)
                    if target_piece and target_piece[0] == side:
                        continue
                        
                    # Save board state
                    old_board = self.board.copy()
                    if self.piece_constrainer(from_square=self.position_to_string(position), 
                                           to_square=self.position_to_string(to_pos), 
                                           piece=piece):
                        # Test if this move would put king in check
                        captured_piece = self.get_piece_at_position(to_pos)
                        self.clear_piece(piece=piece, position=position)
                        if captured_piece:
                            self.clear_piece(piece=captured_piece, position=to_pos)
                        self.set_piece(piece=piece, position=to_pos)
                        if not self.determine_if_checked(side=side):
                            valid_moves.add(to_pos)
                    # Restore board state
                    self.board = old_board
                except:
                    # Restore board state
                    self.board = old_board
                    continue
                    
            # Add castling moves if available
            if side == 'w':
                if self.w_castle_k and not self.get_piece_at_position(61) and not self.get_piece_at_position(62):  # f1 and g1 must be empty
                    # Check if king would pass through check
                    old_board = self.board.copy()
                    try:
                        # Test f1
                        self.clear_piece(piece=piece, position=position)
                        self.set_piece(piece=piece, position=61)
                        if not self.determine_if_checked(side=side):
                            # Test g1
                            self.clear_piece(piece=piece, position=61)
                            self.set_piece(piece=piece, position=62)
                            if not self.determine_if_checked(side=side):
                                valid_moves.add(62)  # g1
                    except:
                        pass
                    self.board = old_board
                    
                if self.w_castle_q and not self.get_piece_at_position(59) and not self.get_piece_at_position(58) and not self.get_piece_at_position(57):  # d1, c1, and b1 must be empty
                    # Check if king would pass through check
                    old_board = self.board.copy()
                    try:
                        # Test d1
                        self.clear_piece(piece=piece, position=position)
                        self.set_piece(piece=piece, position=59)
                        if not self.determine_if_checked(side=side):
                            # Test c1
                            self.clear_piece(piece=piece, position=59)
                            self.set_piece(piece=piece, position=58)
                            if not self.determine_if_checked(side=side):
                                valid_moves.add(58)  # c1
                    except:
                        pass
                    self.board = old_board
            else:
                if self.b_castle_k and not self.get_piece_at_position(5) and not self.get_piece_at_position(6):  # f8 and g8 must be empty
                    # Check if king would pass through check
                    old_board = self.board.copy()
                    try:
                        # Test f8
                        self.clear_piece(piece=piece, position=position)
                        self.set_piece(piece=piece, position=5)
                        if not self.determine_if_checked(side=side):
                            # Test g8
                            self.clear_piece(piece=piece, position=5)
                            self.set_piece(piece=piece, position=6)
                            if not self.determine_if_checked(side=side):
                                valid_moves.add(6)  # g8
                    except:
                        pass
                    self.board = old_board
                    
                if self.b_castle_q and not self.get_piece_at_position(3) and not self.get_piece_at_position(2) and not self.get_piece_at_position(1):  # d8, c8, and b8 must be empty
                    # Check if king would pass through check
                    old_board = self.board.copy()
                    try:
                        # Test d8
                        self.clear_piece(piece=piece, position=position)
                        self.set_piece(piece=piece, position=3)
                        if not self.determine_if_checked(side=side):
                            # Test c8
                            self.clear_piece(piece=piece, position=3)
                            self.set_piece(piece=piece, position=2)
                            if not self.determine_if_checked(side=side):
                                valid_moves.add(2)  # c8
                    except:
                        pass
                    self.board = old_board
            return valid_moves

        # For non-king pieces
        possible_moves_bitboard = lookup_table[position]
        for to_pos in range(64):
            if possible_moves_bitboard & (1 << to_pos):
                try:
                    # Skip if destination has piece of same color
                    target_piece = self.get_piece_at_position(to_pos)
                    if target_piece and target_piece[0] == side:
                        continue
                        
                    # Save board state
                    old_board = self.board.copy()
                    if self.piece_constrainer(from_square=self.position_to_string(position), 
                                           to_square=self.position_to_string(to_pos), 
                                           piece=piece):
                        # Test if this move would leave us in check
                        captured_piece = self.get_piece_at_position(to_pos)
                        self.clear_piece(piece=piece, position=position)
                        if captured_piece:
                            self.clear_piece(piece=captured_piece, position=to_pos)
                        self.set_piece(piece=piece, position=to_pos)
                        if not self.determine_if_checked(side=side):
                            valid_moves.add(to_pos)
                    # Restore board state
                    self.board = old_board
                except:
                    # Restore board state
                    self.board = old_board
                    continue
                    
        return valid_moves
        
    def draw_highlighted_moves(self):
        """
        Draw blue highlights for all valid moves of the selected piece.
        """
        highlight_color = pg.Color(100, 149, 237, 128)  # Semi-transparent cornflower blue
        highlight_surface = pg.Surface((self.square_size, self.square_size), pg.SRCALPHA)
        pg.draw.rect(highlight_surface, highlight_color, highlight_surface.get_rect())
        
        for pos in self.highlighted_moves:
            row = pos // 8
            col = pos % 8
            if self.board_orientation == 'b':  # Only flip if board is oriented from black's perspective
                row = 7 - row
                col = 7 - col
            self.screen.blit(highlight_surface, (col * self.square_size, row * self.square_size))
            
    def highlight_last_computer_move(self):
        """
        Highlights the squares involved in the computer's last move.
        Uses a different color to distinguish from the selected square highlight.
        """
        if not self.last_computer_move:
            return
            
        from_pos, to_pos = self.last_computer_move
        # Convert positions to board coordinates
        from_row, from_col = from_pos // 8, from_pos % 8
        to_row, to_col = to_pos // 8, to_pos % 8
        
        # Adjust for board orientation
        if self.board_orientation == 'b':
            from_row, from_col = 7 - from_row, 7 - from_col
            to_row, to_col = 7 - to_row, 7 - to_col
            
        # Use a different color (yellow) for the computer's move
        highlight_color = (255, 215, 0)  # Gold color
        # Draw slightly thicker borders for better visibility
        pg.draw.rect(self.screen, highlight_color, 
                    (from_col * self.square_size, from_row * self.square_size, 
                     self.square_size, self.square_size), 4)
        pg.draw.rect(self.screen, highlight_color, 
                    (to_col * self.square_size, to_row * self.square_size, 
                     self.square_size, self.square_size), 4)

    def move_piece(self, from_square=None, to_square=None, piece=None):
        """Override move_piece to update PGN moves and handle time control."""
        # Store the current state to check for captures and checks
        is_capture = False
        move_successful = False
        
        # Save the current board state before attempting the move
        old_board = self.board.copy()
        old_white_to_move = self.white_to_move
        old_w_castle_k = self.w_castle_k
        old_w_castle_q = self.w_castle_q
        old_b_castle_k = self.b_castle_k
        old_b_castle_q = self.b_castle_q
        old_illegal_played = self.illegal_played
        
        try:
            # Check for capture before making the move
            if piece not in ["OO", "OOO"] and to_square:  # Only check for capture on non-castling moves
                to_pos = self.string_to_position(to_square)
                is_capture = self.get_piece_at_position(to_pos) is not None
            
            # Make the move
            super().move_piece(from_square, to_square, piece)
            
            # Return immediately if the move was illegal
            if self.illegal_played:
                return
            
            # Handle time control
            if self.time_control:
                current_time = pg.time.get_ticks()
                
                # Update times
                if self.last_move_time is not None:
                    time_since_last_move = current_time - self.last_move_time
                    if not self.white_to_move:  # White just moved
                        self.white_time -= time_since_last_move
                        if self.first_move_made:  # Don't add increment on first move
                            self.white_time += self.increment
                    else:  # Black just moved
                        self.black_time -= time_since_last_move
                        self.black_time += self.increment
                
                self.last_move_time = current_time
                if not self.first_move_made:
                    self.first_move_made = True
            
            print(self.illegal_played)
            # In human vs human mode, flip the board after each successful move
            if self.game_mode == 'human' and not self.illegal_played:
                
                self.board_orientation = 'w' if self.board_orientation == 'b' else 'b'
            
            # Update PGN for successful moves
            if not self.promoting and not self.illegal_played:  # Don't add to PGN yet if promoting
                last_move = self.move_log[-1] if self.move_log.size > 0 else None
                if last_move is not None and last_move[0] != -1:
                    # Handle castling moves
                    if piece in ["OO", "OOO"]:
                        pgn_move = "O-O" if piece == "OO" else "O-O-O"
                        if self.checkmated:
                            pgn_move += '#'
                        elif self.white_in_check or self.black_in_check:
                            pgn_move += '+'
                        self.pgn_moves.append(pgn_move)
                    else:
                        # Get move details for non-castling moves
                        piece_type = list(self.piece_enum.keys())[list(self.piece_enum.values()).index(last_move[0])]
                        from_pos = last_move[1]
                        to_pos = last_move[2]
                        
                        # Convert to PGN and add to list
                        pgn_move = self.convert_to_pgn(
                            piece_type,
                            from_pos,
                            to_pos,
                            is_capture=is_capture,
                            is_check=self.white_in_check or self.black_in_check,
                            is_checkmate=self.checkmated,
                            is_promotion=self.promoting
                        )
                        self.pgn_moves.append(pgn_move)
            
        except Exception as e:
            # Restore the previous state if the move was invalid
            self.board = old_board
            self.white_to_move = old_white_to_move
            self.w_castle_k = old_w_castle_k
            self.w_castle_q = old_w_castle_q
            self.b_castle_k = old_b_castle_k
            self.b_castle_q = old_b_castle_q
            # Don't reset illegal_played flag when restoring state
            print(f"An error occurred while moving the piece: {str(e)}")
            return

    def check_time_out(self):
        """
        Check if either player has run out of time.
        Returns True if a player has lost on time, False otherwise.
        """
        if not self.time_control:
            return False
            
        if self.white_time <= 0:
            self.display_end_game_message('t', 'b')  # Black wins on time
            return True
        elif self.black_time <= 0:
            self.display_end_game_message('t', 'w')  # White wins on time
            return True
        return False
        
    def show_time_control_selection(self):
        """
        Display the time control selection screen and return the selected time control.
        """
        # Colors
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GREEN = (0, 100, 0)
        HOVER_GREEN = (0, 120, 0)
        
        # Create buttons
        button_width = 300
        button_height = 80
        button_x = self.width // 2 - button_width // 2
        title_margin = self.height // 3  # Increased margin for title
        blitz_button_y = title_margin + button_height
        rapid_button_y = blitz_button_y + button_height + 40
        long_button_y = rapid_button_y + button_height + 40
        
        # Font
        pg.font.init()
        font = pg.font.SysFont('Arial', 32)
        title_font = pg.font.SysFont('Arial', 48)
        
        # Title text
        title = title_font.render('Select Time Control', True, BLACK)
        title_rect = title.get_rect(center=(self.width // 2, title_margin - button_height))
        
        while True:
            mouse_pos = pg.mouse.get_pos()
            
            # Check if mouse is hovering over buttons
            blitz_hover = button_x <= mouse_pos[0] <= button_x + button_width and blitz_button_y <= mouse_pos[1] <= blitz_button_y + button_height
            rapid_hover = button_x <= mouse_pos[0] <= button_x + button_width and rapid_button_y <= mouse_pos[1] <= rapid_button_y + button_height
            long_hover = button_x <= mouse_pos[0] <= button_x + button_width and long_button_y <= mouse_pos[1] <= long_button_y + button_height
            
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
                if event.type == pg.MOUSEBUTTONDOWN:
                    if blitz_hover:
                        self.white_time = 5 * 60 * 1000  # 5 minutes in milliseconds
                        self.black_time = 5 * 60 * 1000
                        self.increment = 10 * 1000  # 10 seconds in milliseconds
                        return 'blitz'
                    elif rapid_hover:
                        self.white_time = 15 * 60 * 1000  # 15 minutes in milliseconds
                        self.black_time = 15 * 60 * 1000
                        self.increment = 15 * 1000  # 15 seconds in milliseconds
                        return 'rapid'
                    elif long_hover:
                        self.white_time = 45 * 60 * 1000  # 45 minutes in milliseconds
                        self.black_time = 45 * 60 * 1000
                        self.increment = 30 * 1000  # 30 seconds in milliseconds
                        return 'long'
            
            # Draw screen
            self.screen.fill(WHITE)
            
            # Draw title
            self.screen.blit(title, title_rect)
            
            # Draw buttons
            pg.draw.rect(self.screen, HOVER_GREEN if blitz_hover else GREEN, (button_x, blitz_button_y, button_width, button_height))
            pg.draw.rect(self.screen, HOVER_GREEN if rapid_hover else GREEN, (button_x, rapid_button_y, button_width, button_height))
            pg.draw.rect(self.screen, HOVER_GREEN if long_hover else GREEN, (button_x, long_button_y, button_width, button_height))
            
            # Draw button text
            blitz_text = font.render('Blitz (5+10)', True, WHITE)
            rapid_text = font.render('Rapid (15+15)', True, WHITE)
            long_text = font.render('Long (45+30)', True, WHITE)
            
            blitz_text_rect = blitz_text.get_rect(center=(self.width // 2, blitz_button_y + button_height // 2))
            rapid_text_rect = rapid_text.get_rect(center=(self.width // 2, rapid_button_y + button_height // 2))
            long_text_rect = long_text.get_rect(center=(self.width // 2, long_button_y + button_height // 2))
            
            self.screen.blit(blitz_text, blitz_text_rect)
            self.screen.blit(rapid_text, rapid_text_rect)
            self.screen.blit(long_text, long_text_rect)
            
            pg.display.update()

    def draw_clock(self):
        """
        Draw the chess clock for both players.
        """
        font = pg.font.SysFont('Arial', 36)
        padding = 20
        
        # Calculate remaining time for both players
        if self.last_move_time is not None and self.first_move_made:
            current_time = pg.time.get_ticks()
            time_since_last_move = current_time - self.last_move_time
            if self.white_to_move:
                self.white_time -= time_since_last_move
            else:
                self.black_time -= time_since_last_move
            self.last_move_time = current_time
        
        # Format times
        white_minutes = max(0, int(self.white_time // 60000))
        white_seconds = max(0, int((self.white_time % 60000) // 1000))
        black_minutes = max(0, int(self.black_time // 60000))
        black_seconds = max(0, int((self.black_time % 60000) // 1000))
        
        # Create text surfaces
        white_text = font.render(f'{white_minutes:02d}:{white_seconds:02d}', True, pg.Color('Black'))
        black_text = font.render(f'{black_minutes:02d}:{black_seconds:02d}', True, pg.Color('Black'))
        
        # Create background boxes with transparency
        box_padding = 10
        box_width = max(white_text.get_width(), black_text.get_width()) + 2 * box_padding
        box_height = white_text.get_height() + 2 * box_padding
        
        # Position the clocks at top and bottom right
        black_rect = black_text.get_rect(topright=(self.width - padding - box_padding, padding + box_padding))
        white_rect = white_text.get_rect(bottomright=(self.width - padding - box_padding, self.height - padding - box_padding))
        
        black_box = pg.Rect(self.width - padding - box_width, padding, box_width, box_height)
        white_box = pg.Rect(self.width - padding - box_width, self.height - padding - box_height, box_width, box_height)
        
        # Create semi-transparent surfaces
        box_surface = pg.Surface((box_width, box_height), pg.SRCALPHA)
        pg.draw.rect(box_surface, (255, 255, 255, 180), box_surface.get_rect())  # White with 180/255 alpha
        
        # Draw backgrounds with transparency
        self.screen.blit(box_surface, black_box)
        self.screen.blit(box_surface, white_box)
        
        # Draw borders
        pg.draw.rect(self.screen, pg.Color('Black'), black_box, 2)
        pg.draw.rect(self.screen, pg.Color('Black'), white_box, 2)
        
        # Draw text
        self.screen.blit(black_text, black_rect)
        self.screen.blit(white_text, white_rect)
        
    def create_move_window(self):
        """Create a separate window for displaying moves in PGN format."""
        self.move_window = pg.display.set_mode((self.width + self.move_window_width, self.height))
        self.move_font = pg.font.SysFont('Arial', 20)
        self.move_surface = pg.Surface((self.move_window_width, self.height))
        pg.display.set_caption("PythonicChess - Game and Moves")

    def convert_to_pgn(self, piece, from_pos, to_pos, is_capture=False, is_check=False, is_checkmate=False, is_promotion=False, promotion_piece=None):
        """Convert a move to PGN notation."""
        files = 'abcdefgh'
        ranks = '87654321'
        
        # Handle castling
        if piece == "OO":
            return "O-O" + ('#' if is_checkmate else '+' if is_check else '')
        elif piece == "OOO":
            return "O-O-O" + ('#' if is_checkmate else '+' if is_check else '')
            
        # Get piece symbol
        piece_symbols = {'P': '', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K'}
        piece_type = piece[1] if isinstance(piece, str) and len(piece) > 1 else piece
        piece_symbol = piece_symbols[piece_type]
        
        # Get square coordinates
        from_file = files[from_pos % 8]
        from_rank = ranks[from_pos // 8]
        to_file = files[to_pos % 8]
        to_rank = ranks[to_pos // 8]
        
        # Build move string
        move = ''
        if piece_type != 'P':
            move += piece_symbol
        if is_capture:
            if piece_type == 'P':
                move += from_file
            move += 'x'
        move += to_file + to_rank
        
        # Add promotion if applicable
        if is_promotion and promotion_piece:
            move += '=' + promotion_piece
            
        # Add check/checkmate symbol
        if is_checkmate:
            move += '#'
        elif is_check:
            move += '+'
            
        return move

    def update_move_display(self):
        """Update the move window with the current PGN moves."""
        if not self.move_window:
            return
            
        # Clear the move surface
        self.move_surface.fill(pg.Color('white'))
        
        # Draw title
        title_font = pg.font.SysFont('Arial', 24, bold=True)
        title = title_font.render('Move History', True, pg.Color('black'))
        title_rect = title.get_rect(centerx=self.move_window_width // 2, top=10)
        self.move_surface.blit(title, title_rect)
        
        # Draw moves
        y = 50 + self.scroll_y
        line_height = 30
        for i in range(0, len(self.pgn_moves), 2):
            # Move number
            move_num = i // 2 + 1
            num_text = self.move_font.render(f"{move_num}.", True, pg.Color('black'))
            self.move_surface.blit(num_text, (10, y))
            
            # White's move
            white_text = self.move_font.render(self.pgn_moves[i], True, pg.Color('black'))
            self.move_surface.blit(white_text, (50, y))
            
            # Black's move if it exists
            if i + 1 < len(self.pgn_moves):
                black_text = self.move_font.render(self.pgn_moves[i + 1], True, pg.Color('black'))
                self.move_surface.blit(black_text, (150, y))
            
            y += line_height
            
        # Update max scroll value
        self.max_scroll = min(0, self.height - (len(self.pgn_moves) // 2 + 1) * line_height - 50)
        
        # Draw scroll bar if needed
        if self.max_scroll < 0:
            scroll_height = self.height * (self.height / (abs(self.max_scroll) + self.height))
            scroll_pos = (-self.scroll_y / self.max_scroll) * (self.height - scroll_height)
            pg.draw.rect(self.move_surface, pg.Color('gray'), 
                        (self.move_window_width - 15, scroll_pos, 10, scroll_height))
        
        # Draw border
        pg.draw.line(self.move_surface, pg.Color('black'), (0, 0), (0, self.height), 2)

    def handle_scroll(self, event):
        """Handle mouse wheel scrolling in the move window."""
        if event.type == pg.MOUSEWHEEL:
            mouse_x = pg.mouse.get_pos()[0]
            if mouse_x > self.width:  # Only scroll if mouse is in move window
                self.scroll_y = max(min(0, self.scroll_y + event.y * 20), self.max_scroll)

    def draw_game_state(self):
        """Draw the complete game state including the move window."""
        # Create move window if it doesn't exist
        if not self.move_window:
            self.create_move_window()
        
        # Draw the chess board on the left side
        self.draw_board()
        self.draw_highlighted_moves()
        self.draw_pieces()
        if self.highlighted_square:
            self.highlight_square(self.highlighted_square)
        if self.last_computer_move and self.game_mode == 'computer':
            self.highlight_last_computer_move()
        if self.time_control:
            self.draw_clock()
        
        # Update and draw the move window on the right side
        self.update_move_display()
        self.move_window.blit(self.screen, (0, 0))
        self.move_window.blit(self.move_surface, (self.width, 0))

if __name__ == "__main__":
    game = GameInterface()
    signal.signal(signal.SIGINT, signal_handler)
    game.run()