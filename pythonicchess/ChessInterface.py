from ChessLogic import ConstrainedGameState
import numpy as np
import pygame as pg
import sys
import signal
from time import sleep
from ChessEngine import GameEngine
from copy import deepcopy

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
        if not self.white_to_move:
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

    def make_computer_move(self):
        """
        Calculate and make the computer's move using the chess engine.
        """
        if not self.white_to_move and not (self.checkmated or self.drawn):
            self.computer_thinking = True
            print("\n=== Debug: Starting computer move calculation ===")
            print(f"Interface board type before sync: {type(self.board)}")
            print(f"Interface board keys before sync: {list(self.board.keys())}")
            
            # Draw the board with a "thinking" indicator
            self.draw_game_state()
            self.display_thinking_message()
            pg.display.update()
            
            # Sync all necessary game state with the engine
            self.engine.board = deepcopy(self.board)
            print(f"Engine board type after sync: {type(self.engine.board)}")
            print(f"Engine board keys after sync: {list(self.engine.board.keys())}")
            
            self.engine.white_to_move = self.white_to_move
            self.engine.w_castle_k = self.w_castle_k
            self.engine.w_castle_q = self.w_castle_q
            self.engine.b_castle_k = self.b_castle_k
            self.engine.b_castle_q = self.b_castle_q
            self.engine.move_log = self.move_log.copy()
            self.engine.board_states = self.board_states.copy()
            
            print("\n=== Debug: About to call minmax ===")
            print(f"Board type being passed to minmax: {type(self.engine.board)}")
            if not isinstance(self.engine.board, dict):
                print(f"WARNING: Board is not a dictionary! It is: {self.engine.board}")
            
            # Get the best move from the engine
            try:
                _, best_move = self.engine.minmax(self.engine.board, depth=3, alpha=float('-inf'), beta=float('inf'), maximizing_player=False)
                print("\n=== Debug: Minmax completed successfully ===")
            except Exception as e:
                print(f"\n=== Debug: Error in minmax: {str(e)} ===")
                print("Stack trace:")
                import traceback
                traceback.print_exc()
                self.computer_thinking = False
                return
            
            if best_move:
                print(f"\n=== Debug: Best move found: {best_move} ===")
                piece, from_pos, to_pos, promotion_piece = best_move
                if promotion_piece:
                    self.promoted_piece = 'b' + promotion_piece
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
        Display a "Computer is thinking..." message on the screen.
        """
        font = pg.font.SysFont('Arial', 36)
        text = font.render('Computer is thinking...', True, pg.Color('Red'))
        text_rect = text.get_rect(center=(self.width // 2, 30))
        self.screen.blit(text, text_rect)
    
    def run(self):
        """
        The run function initializes the game, handles user input, and updates the game state.
        """
        pg.init()
        clock = pg.time.Clock()
        
        # Show game mode selection screen
        self.game_mode = self.show_game_mode_selection()
        
        running = True
        self.highlighted_square = None
        try:
            while running:
                if self.reset:
                    self.reset_game()
                    # Show game mode selection again after reset
                    self.game_mode = self.show_game_mode_selection()
                
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
                    elif e.type == pg.MOUSEBUTTONDOWN and (self.game_mode == 'human' or self.white_to_move):
                        try:
                            mouse_x, mouse_y = pg.mouse.get_pos()
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
        """
        self.draw_board()  # Draw the base board first
        self.draw_highlighted_moves()  # Draw move highlights
        self.draw_pieces()  # Draw pieces on top
        if self.highlighted_square:  # Draw selection highlight last
            self.highlight_square(self.highlighted_square)
            
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
        """
        for piece, bitboard in self.board.items():
            for i in range(64):
                if (bitboard >> i) & 1:  # if the i-th bit is set
                    row = i // 8
                    col = i % 8
                    if not self.white_to_move:
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
                                        # Add the en passant capture square to valid moves
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
                        
                    if self.piece_constrainer(from_square=self.position_to_string(position), 
                                           to_square=self.position_to_string(to_pos), 
                                           piece=piece):
                        valid_moves.add(to_pos)
                except:
                    continue
                    
            # Add castling moves if available
            if side == 'w':
                if self.w_castle_k and not self.get_piece_at_position(61) and not self.get_piece_at_position(62):  # f1 and g1 must be empty
                    valid_moves.add(62)  # g1
                if self.w_castle_q and not self.get_piece_at_position(59) and not self.get_piece_at_position(58) and not self.get_piece_at_position(57):  # d1, c1, and b1 must be empty
                    valid_moves.add(58)  # c1
            else:
                if self.b_castle_k and not self.get_piece_at_position(5) and not self.get_piece_at_position(6):  # f8 and g8 must be empty
                    valid_moves.add(6)   # g8
                if self.b_castle_q and not self.get_piece_at_position(3) and not self.get_piece_at_position(2) and not self.get_piece_at_position(1):  # d8, c8, and b8 must be empty
                    valid_moves.add(2)   # c8
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
                        
                    if self.piece_constrainer(from_square=self.position_to_string(position), 
                                           to_square=self.position_to_string(to_pos), 
                                           piece=piece):
                        valid_moves.add(to_pos)
                except:
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
            if not self.white_to_move:
                row = 7 - row
                col = 7 - col
            self.screen.blit(highlight_surface, (col * self.square_size, row * self.square_size))
            
if __name__ == "__main__":
    game = GameInterface()
    signal.signal(signal.SIGINT, signal_handler)
    game.run()