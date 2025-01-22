"""
This is the chess engine module. It will be responsible for 
move generation and evaluation of chess games.
"""
import warnings
from copy import deepcopy
import numpy as np

import sys
import hashlib
import traceback

def print_bits(x, n_bits=8):
    # Create a format string for the specified number of bits with leading zeros
    format_string = '{:0' + str(n_bits) + 'b}'
    # Format the integer to its binary representation with the specified number of bits
    binary_representation = format_string.format(x)
    # Print the binary representation
    print(binary_representation)
    # Return the binary representation as a list of bits
    return [int(bit) for bit in binary_representation]
def bitscan_forward(bb):
    assert bb != 0  # Precondition: bb should not be zero
    bb ^= bb - 1
    t32 = (bb ^ (bb >> 32)) & 0xFFFFFFFF
    t32 ^= t32 >> 16
    t32 ^= t32 >> 8
    t32 += t32 >> 16
    t32 -= (t32 >> 8) + 51
    LSB_64_table = [0, 47, 1, 56, 48, 27, 2, 60, 57, 49, 41, 37, 28, 16, 3, 61, 54, 58, 35, 52, 50, 42, 21, 44, 38, 32, 29, 23, 17, 11, 4, 62, 46, 55, 26, 59, 40, 36, 15, 53, 34, 51, 20, 43, 31, 22, 10, 45, 25, 39, 14, 33, 19, 24, 13, 30, 18, 12, 5, 63]
    return LSB_64_table[t32 & 255]
class InvalidRookMoveError(Exception):
    def __init__(self, message="An invalid rook move was attempted."):
        self.message = message
        super().__init__(self.message)

class InvalidBishopMoveError(Exception):


    def __init__(self, message="An invalid bishop move was attempted."):
        self.message = message
        super().__init__(self.message)


class InvalidQueenMoveError(Exception):
    def __init__(self, message="An invalid queen move was attempted."):
        self.message = message
        super().__init__(self.message)
class GameState:
    """
    This class is used to store the current state of the game.
    """
    def __init__(self):
        self.board = {'wP': 0, 'wR': 0, 'wN': 0, 'wB': 0, 'wQ': 0, 'wK': 0,
                      'bP': 0, 'bR': 0, 'bN': 0, 'bB': 0, 'bQ': 0, 'bK': 0}
        self.piece_enum = {piece: i for i, piece in enumerate(self.board)}
        self.piece_enum["wOO"] = -2
        self.piece_enum["wOOO"] = -3
        self.piece_enum["bOO"] = -4
        self.piece_enum["bOOO"] = -5
        self.piece_enum["=wQ"] = -64
        self.piece_enum["=wR"] = -65
        self.piece_enum["=wN"] = -66
        self.piece_enum["=wB"] = -67
        self.piece_enum["=bQ"] = -68
        self.piece_enum["=bR"] = -69
        self.piece_enum["=bN"] = -70
        self.piece_enum["=bB"] = -71
        
        self.files = ["a", "b", "c", "d", "e", "f", "g", "h"]
        self.ranks = [1, 2, 3, 4, 5, 6, 7, 8]
        
        
        self.squares = [f + str(r) for f in self.files for r in self.ranks]
        #self.squares.reverse()
        self.first_move = True
        self.move_log = np.array([[-1, -1, -1]], dtype=np.int8)
        self.white_to_move = True
        self.illegal_played = False
        
        
    def set_piece(self, piece, position):
        # Determine the color of the piece
        color = 'w' if 'w' in piece else 'b'
        
        # Check if any piece of the same color is already at the position
        for p in self.board:
            if p.startswith(color) and (self.board[p] & (1 << position)):
                raise ValueError(f"Position {position} is already occupied by a piece of the same color.")
        # Get the opposite color
        opposite_color = 'b' if color == 'w' else 'w'
        # Check if any piece of the opposite color is at the position, and clear it if it is not a king
        for p in self.board:
            if p.startswith(opposite_color) and (self.board[p] & (1 << position)) and not p.endswith('K'):
                self.clear_piece(p, position)
        # Check if the opposite colored king is at the position
        if self.board[opposite_color + 'K'] & (1 << position):
            raise ValueError(f"A king cannot be captured: {opposite_color}K at position {position}")
        
        # Set the piece if the position is not occupied
        self.board[piece] |= 1 << position

    def clear_piece(self, piece, position):
        self.board[piece] &= ~(1 << position)
        
    def get_piece_at_square(self, square):
        for piece, bitboard in self.board.items():
            if bitboard & (1 << self.string_to_position(square)):
                return piece
        return None
    
    def get_piece_at_position(self, position):
        for piece, bitboard in self.board.items():
            if bitboard & (1 << position):
                return piece
        return None
    

    def set_start_position(self):
        # Initialize black pieces
        self.board['bP'] = 0xFF << 8
        self.board['bR'] = 0x81
        self.board['bN'] = 0x42
        self.board['bB'] = 0x24
        self.board['bK'] = 0x10
        self.board['bQ'] = 0x8

        # Initialize white pieces
        self.board['wP'] = 0xFF << 48
        self.board['wR'] = 0x81 << 56
        self.board['wN'] = 0x42 << 56
        self.board['wB'] = 0x24 << 56
        self.board['wK'] = 0x10 << 56
        self.board['wQ'] = 0x8 << 56
        self.white_to_move = True
        self.move_log = np.array([[-1, -1, -1]], dtype=np.int8)
        

    
    def get_board(self):
        return self.board
    
    def clear_board(self):
        self.board = {'wP': 0, 'wR': 0, 'wN': 0, 'wB': 0, 'wQ': 0, 'wK': 0,
                      'bP': 0, 'bR': 0, 'bN': 0, 'bB': 0, 'bQ': 0, 'bK': 0}
        self.move_log = np.array([[-1, -1, -1]], dtype=np.int8)

    def string_to_position(self, square_str):
        """
        Convert a chess square string to a position index on the board.

        Args:
            square_str (str): A string representing a chess square, e.g. 'a1' or 'h8'.

        Returns:
            int: The position index of the square on the board.
        Raises:
            ValueError: If the input square string is invalid.
        """
        if len(square_str) != 2:
            raise ValueError("Invalid square string")
        file, rank = square_str[0], square_str[1]
        if file not in self.files or not rank.isdigit() or int(rank) not in self.ranks:
            raise ValueError("Invalid square string")

        rank_index = self.ranks.index(int(rank))
        file_index = self.files.index(file)
        position = (7 - rank_index) * 8 + file_index
        return position
    
    def position_to_string(self, position):
        """
        Converts a position index to a string representation of the file and rank. 

        Parameters:
            position (int): The position index to be converted.

        Returns:
            str: The string representation of the file and rank.
        """
        file_index = position % 8
        rank_index = 7 - position // 8
        file = self.files[file_index]
        rank = str(self.ranks[rank_index])
        return file + rank
    
    def swap_piece_mapping(self, piece_to_char):
        new_map = {k.replace('w', 'temp').replace('b', 'w').replace('temp', 'b'): v for k, v in piece_to_char.items()}
        return new_map
    
    def get_board_representation(self):
        """
        Returns the board representation as a 2D numpy array. The array represents the current state of the chess board with pieces and empty squares. The piece_to_char dictionary maps piece codes to their corresponding Unicode chess piece characters. The final representation is flipped vertically if it's white's turn to move. Returns:
        - board_representation: a 2D numpy array representing the current state of the chess board.
        """
        board_representation = np.full((8, 8), '.', dtype=str)
        piece_to_char = {
            'wR': '♜', 'wN': '♞', 'wB': '♝', 'wQ': '♛', 'wK': '♚', 'wP': '♟︎',
            'bR': '♖', 'bN': '♘', 'bB': '♗', 'bQ': '♕', 'bK': '♔', 'bP': '♙'
        }
        #piece_to_char = self.swap_piece_mapping(piece_to_char)
        
        for piece, bitboard in self.board.items():
            for i in range(64):
                if bitboard & (1 << i):
                    rank = 7 - (i // 8)
                    file = i % 8
                    board_representation[rank, file] = piece_to_char[piece]
        
        if self.white_to_move:
            board_representation = np.flip(board_representation, axis=0)
        
        return board_representation
    
    def file_check(self, square_str):
        if len(square_str) != 2 or square_str[0] not in self.files or not square_str[1].isdigit() or int(square_str[1]) not in self.ranks:
            raise ValueError("Invalid square string")

        #file_index = self.files.index(square_str[0])
        file_occupancy = 0

        for rank in self.ranks:
            position = self.string_to_position(f"{square_str[0]}{rank}")
            for piece, bitboard in self.board.items():
                if bitboard & (1 << position):
                    file_occupancy |= (1 << (rank - 1))
                    break

        return file_occupancy
    
    def rank_check(self, square_str):
        if len(square_str) != 2 or square_str[0] not in self.files or not square_str[1].isdigit() or int(square_str[1]) not in self.ranks:
            raise ValueError("Invalid square string")

        #rank_index = self.ranks.index(int(square_str[1]))
        rank_occupancy = 0

        for file in self.files:
            position = self.string_to_position(f"{file}{square_str[1]}")
            for piece, bitboard in self.board.items():
                if bitboard & (1 << position):
                    rank_occupancy |= (1 << (self.files.index(file)))
                    break

        return rank_occupancy
    
    def diag_check(self, square_str):
        if len(square_str) != 2 or square_str[0] not in self.files or not square_str[1].isdigit() or int(square_str[1]) not in self.ranks:
            raise ValueError("Invalid square string")

        position = self.string_to_position(square_str)
        major_diag_mask = 0
        minor_diag_mask = 0

        # Calculate masks for the major and minor diagonals
        for rank in range(8):
            for file in range(8):
                if rank - file == (7 - position // 8) - (position % 8):
                    major_diag_mask |= (1 << ((7 - rank) * 8 + file))
                if rank + file == (7 - position // 8) + (position % 8):
                    minor_diag_mask |= (1 << ((7 - rank) * 8 + file))

        # Check for occupancy using the masks
        major_diag_occupancy = 0
        minor_diag_occupancy = 0
        for piece, bitboard in self.board.items():
            major_diag_occupancy |= (bitboard & major_diag_mask)
            minor_diag_occupancy |= (bitboard & minor_diag_mask)

        return major_diag_occupancy, minor_diag_occupancy
    
    def move_piece(self, from_square="", to_square="", piece=None):
        """
        Move a piece from one square to another.
        """
        # Reset illegal_played at the start of each move attempt
        self.illegal_played = False
        
        # Don't proceed if we don't have both squares for any move
        if from_square and not to_square:
            print("Waiting for destination square...")
            return

        old_self = ConstrainedGameState()
        old_board = self.board.copy()
        for attr, value in vars(self).items():
            if isinstance(value, (bool, list, np.ndarray)):
                setattr(old_self, attr, deepcopy(value))
                
        # Convert single letter piece to full piece name with color
        if piece is not None and len(piece) == 1:
            if self.white_to_move:
                piece = 'w' + piece
            else:
                piece = 'b' + piece
                
        # Early check for correct turn
        if piece and not "O" in piece:  # Skip this check for castling notation (OO/OOO)
            is_white_piece = piece.startswith('w')
            if (is_white_piece and not self.white_to_move) or (not is_white_piece and self.white_to_move):
                self.illegal_played = True
                raise ValueError(f"Not {piece[0]}'s turn to move")
                
        try:
            if self.piece_castle_conditional(piece):
                from_position = self.string_to_position(from_square)
                to_position = self.string_to_position(to_square)
                moving_piece = None
                for p, bitboard in self.board.items():
                    if bitboard & (1 << from_position):
                        if piece is None or p == piece:
                            moving_piece = p
                            break
                if moving_piece is None:
                    self.illegal_played = True
                    raise ValueError(f"No {piece} at {from_square}.")
                    
                promotion = False
                if 'P' in moving_piece and ((moving_piece.startswith('w') and to_position // 8 == 0) or (moving_piece.startswith('b') and to_position // 8 == 7)):
                    promotion = True
                    self.promoting = True
                else:
                    self.promoting = False
                    
                if self.piece_constrainer(from_square, to_square, piece=moving_piece):
                    # Handle castling move
                    if moving_piece and moving_piece[1] == 'K' and abs(from_position % 8 - to_position % 8) == 2:
                        direction = "k" if to_position % 8 > from_position % 8 else "q"
                        side = moving_piece[0]
                        self.castling_move(side=side, direction=direction)
                        self.log_castle(side + ("OO" if direction == "k" else "OOO"))
                    else:
                        # Handle regular move
                        self.clear_piece(moving_piece, from_position)
                        if not promotion:
                            self.set_piece(moving_piece, to_position)
                        if promotion:
                            self.promoting_from_pos = from_position
                            self.promoting_to_pos = to_position
                        
                    self.king_or_rook_moved(moving_piece, from_square=from_square)

                    
                    self.update_board_state()
                    
                    # Check if the move was legal by verifying check status
                    white_check = self.determine_if_checked(side="w")
                    black_check = self.determine_if_checked(side="b")
                    
                    # A move is illegal if it leaves or puts your own king in check
                    if self.white_to_move and white_check:
                        self.illegal_played = True
                        raise ValueError("Illegal move. White is still in check or is put in check.")

                    if not self.white_to_move and black_check:
                        self.illegal_played = True
                        raise ValueError("Illegal move. Black is still in check or is put in check.")
                    
                    # Update check status and evaluate checkmate
                    self.white_in_check = bool(white_check)
                    self.black_in_check = bool(black_check)
                    
                    # Toggle turn
                    self.white_to_move = not self.white_to_move
                    
                    # Now check for checkmate/stalemate with the new turn state
                    self.assign_checking_and_final_states()
                    
                    if not self.promoting:
                        move_array = np.array([self.piece_enum[moving_piece], from_position, to_position], dtype=np.int8)
                        self.move_log = np.vstack([self.move_log, move_array])
                    if self.first_move:
                        self.move_log = self.move_log[1:]
                        self.first_move = False
                else:
                    self.illegal_played = True
                    raise ValueError(f"Illegal move from {from_square} to {to_square} for {moving_piece}.")
            else:
                if self.piece_constrainer(piece=piece):
                    self.king_or_rook_moved(moving_piece=piece)
                    self.castling_logic(piece)
                    self.update_board_state()
                    
        except Exception as e:
            # Save illegal_played state
            was_illegal = self.illegal_played
            # Restore the previous state if the move was invalid
            for attr in vars(old_self):
                setattr(self, attr, getattr(old_self, attr))
            self.board = old_board
            # Restore illegal_played to its value after the error
            self.illegal_played = was_illegal
            print(f"An error occurred while moving the piece: {e}")
            traceback.print_exc(file=sys.stdout)
            return
    def display_board(self):
        print(self.get_board_representation())
        
class ConstrainedGameState(GameState):
    def __init__(self):
        super().__init__()
        self.board_states = {}
        self.N_lookup = self.get_all_possible_knight_moves()
        self.B_lookup = self.get_all_possible_bishop_moves()
        self.R_lookup = self.get_all_possible_rook_moves()
        self.Q_lookup = self.get_all_possible_queen_moves()
        
        self.editing = False
        self.w_castle_k = True
        self.w_castle_q = True
        self.b_castle_k = True
        self.b_castle_q = True
        
        self.white_king_moved = False
        self.black_king_moved = False
        self.white_in_check = False
        self.black_in_check = False
        
        self.KR_moved_white = False # Has the kingside rook moved for white
        self.QR_moved_white = False # Has the queenside rook moved for white
        
        self.KR_moved_black = False # Has the kingside rook moved for black
        self.QR_moved_black = False # Has the queenside rook moved for black
        self.castle_mapping = {"OO": "k",
                               "OOO": "q"}
        self.castle_logmap = {"wOO": np.array([-2, -2, -2]),
                              "wOOO": np.array([-3, -3, -3]),
                              "bOO": np.array([-4, -4, -4]),
                              "bOOO": np.array([-5, -5, -5])}
        self.checkmated = False
        self.drawn = False
        self.en_passant_target = None
        
        self.promoting = False
        self.promoted_piece = None
        self.promoting_from_pos = None
        self.promoting_to_pos = None
        
    
    def hash_board_state(self):
        board_string = ''.join(str(self.board[piece]) for piece in sorted(self.board))
        return hashlib.sha256(board_string.encode()).hexdigest()
    
    def update_board_state(self):
        board_hash = self.hash_board_state()
        if board_hash in self.board_states:
            self.board_states[board_hash] += 1
        else:
            self.board_states[board_hash] = 1
    
    def get_all_possible_knight_moves(self, board=None):
        board = board if board is not None else self.board
        knight_moves = {}
        for i in range(64):
            x, y = divmod(i, 8)
            
            # Calculate all possible knight moves for this position (for lookup table)
            possible_moves_bitboard = 0
            knight_offsets = [
                (-2, -1), (-2, 1),  # Up 2, left/right 1
                (2, -1), (2, 1),    # Down 2, left/right 1
                (-1, -2), (-1, 2),  # Up 1, left/right 2
                (1, -2), (1, 2)     # Down 1, left/right 2
            ]
            
            # First calculate all theoretically possible moves for this position
            for dx, dy in knight_offsets:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < 8 and 0 <= new_y < 8:
                    target_pos = new_x * 8 + new_y
                    possible_moves_bitboard |= 1 << target_pos
            
            # Store all theoretical moves in lookup (needed for checking)
            knight_moves[i] = possible_moves_bitboard
            
            # Now filter moves for actual knights on the board
            for side in ['w', 'b']:
                if board.get(side + 'N', 0) & (1 << i):
                    # Get own pieces to avoid moving to squares occupied by own pieces
                    own_pieces = 0
                    for piece in ['P', 'R', 'N', 'B', 'Q', 'K']:
                        own_pieces |= board.get(side + piece, 0)
                    
                    # Filter out moves to squares occupied by own pieces
                    knight_moves[i] &= ~own_pieces
        
        return knight_moves
    
    def get_all_possible_bishop_moves(self, board=None):
        board = board if board is not None else self.board
        bishop_moves = {}
        
        # Pre-calculate piece positions
        white_pieces = sum(board.get('w' + piece, 0) for piece in ['P', 'R', 'N', 'B', 'Q', 'K'])
        black_pieces = sum(board.get('b' + piece, 0) for piece in ['P', 'R', 'N', 'B', 'Q', 'K'])
        
        # Calculate moves for all squares (needed for checking)
        for i in range(64):
            x, y = divmod(i, 8)
            possible_moves_bitboard = 0
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            
            # Calculate theoretical moves
            for dx, dy in directions:
                curr_x, curr_y = x + dx, y + dy
                while 0 <= curr_x < 8 and 0 <= curr_y < 8:
                    target_pos = curr_x * 8 + curr_y
                    possible_moves_bitboard |= 1 << target_pos
                    curr_x += dx
                    curr_y += dy
            
            # Store theoretical moves
            bishop_moves[i] = possible_moves_bitboard
            
            # Filter moves for actual bishops
            for side, own_pieces, opp_pieces in [('w', white_pieces, black_pieces), 
                                               ('b', black_pieces, white_pieces)]:
                if board.get(side + 'B', 0) & (1 << i):
                    filtered_moves = 0
                    for dx, dy in directions:
                        curr_x, curr_y = x + dx, y + dy
                        while 0 <= curr_x < 8 and 0 <= curr_y < 8:
                            target_pos = curr_x * 8 + curr_y
                            target_bit = 1 << target_pos
                            
                            if target_bit & (own_pieces | opp_pieces):
                                if target_bit & opp_pieces:
                                    filtered_moves |= target_bit
                                break
                            
                            filtered_moves |= target_bit
                            curr_x += dx
                            curr_y += dy
                    
                    bishop_moves[i] = filtered_moves
        
        return bishop_moves

    def get_all_possible_rook_moves(self, board=None):
        board = board if board is not None else self.board
        rook_moves = {}
        
        # Pre-calculate piece positions
        white_pieces = sum(board.get('w' + piece, 0) for piece in ['P', 'R', 'N', 'B', 'Q', 'K'])
        black_pieces = sum(board.get('b' + piece, 0) for piece in ['P', 'R', 'N', 'B', 'Q', 'K'])
        
        # Calculate moves for all squares (needed for checking)
        for i in range(64):
            x, y = divmod(i, 8)
            possible_moves_bitboard = 0
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, left, down, up
            
            # Calculate theoretical moves
            for dx, dy in directions:
                curr_x, curr_y = x + dx, y + dy
                while 0 <= curr_x < 8 and 0 <= curr_y < 8:
                    target_pos = curr_x * 8 + curr_y
                    possible_moves_bitboard |= 1 << target_pos
                    curr_x += dx
                    curr_y += dy
            
            # Store theoretical moves
            rook_moves[i] = possible_moves_bitboard
            
            # Filter moves for actual rooks
            for side, own_pieces, opp_pieces in [('w', white_pieces, black_pieces), 
                                               ('b', black_pieces, white_pieces)]:
                if board.get(side + 'R', 0) & (1 << i):
                    filtered_moves = 0
                    for dx, dy in directions:
                        curr_x, curr_y = x + dx, y + dy
                        while 0 <= curr_x < 8 and 0 <= curr_y < 8:
                            target_pos = curr_x * 8 + curr_y
                            target_bit = 1 << target_pos
                            
                            if target_bit & (own_pieces | opp_pieces):
                                if target_bit & opp_pieces:
                                    filtered_moves |= target_bit
                                break
                            
                            filtered_moves |= target_bit
                            curr_x += dx
                            curr_y += dy
                    
                    rook_moves[i] = filtered_moves
        
        return rook_moves

    def get_all_possible_queen_moves(self, board=None):
        board = board if board is not None else self.board
        queen_moves = {}
        
        # Pre-calculate piece positions
        white_pieces = sum(board.get('w' + piece, 0) for piece in ['P', 'R', 'N', 'B', 'Q', 'K'])
        black_pieces = sum(board.get('b' + piece, 0) for piece in ['P', 'R', 'N', 'B', 'Q', 'K'])
        
        # Calculate moves for all squares (needed for checking)
        for i in range(64):
            x, y = divmod(i, 8)
            possible_moves_bitboard = 0
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0),  # Rook directions
                         (-1, -1), (-1, 1), (1, -1), (1, 1)] # Bishop directions
            
            # Calculate theoretical moves
            for dx, dy in directions:
                curr_x, curr_y = x + dx, y + dy
                while 0 <= curr_x < 8 and 0 <= curr_y < 8:
                    target_pos = curr_x * 8 + curr_y
                    possible_moves_bitboard |= 1 << target_pos
                    curr_x += dx
                    curr_y += dy
            
            # Store theoretical moves
            queen_moves[i] = possible_moves_bitboard
            
            # Filter moves for actual queens
            for side, own_pieces, opp_pieces in [('w', white_pieces, black_pieces), 
                                               ('b', black_pieces, white_pieces)]:
                if board.get(side + 'Q', 0) & (1 << i):
                    filtered_moves = 0
                    for dx, dy in directions:
                        curr_x, curr_y = x + dx, y + dy
                        while 0 <= curr_x < 8 and 0 <= curr_y < 8:
                            target_pos = curr_x * 8 + curr_y
                            target_bit = 1 << target_pos
                            
                            if target_bit & (own_pieces | opp_pieces):
                                if target_bit & opp_pieces:
                                    filtered_moves |= target_bit
                                break
                            
                            filtered_moves |= target_bit
                            curr_x += dx
                            curr_y += dy
                    
                    queen_moves[i] = filtered_moves
        
        return queen_moves
    def convert_to_numpy_arrays(self, lookup):
        for key in lookup:
            lookup[key] = np.array(lookup[key])
        return lookup
    def can_any_piece_move(self, side):
        # Check for bishops, knights, rooks, and queens of the given side
        pieces = ['B', 'N', 'R', 'Q']
        for piece_symbol in pieces:
            piece_key = side + piece_symbol
            bitboard = self.board[piece_key]
            for position in range(64):
                if bitboard & (1 << position):
                    # Get the lookup table for the piece
                    lookup_table = getattr(self, piece_symbol + '_lookup')
                    possible_moves_bitboard = lookup_table[position]
                    for move in range(64):
                        # Check if the move is in the possible moves bitboard
                        if possible_moves_bitboard & (1 << move):
                            # Check if the move is to an unoccupied square
                            if not self.is_square_occupied_by_side(move, side):
                                # Check if the move follows the constraints for the piece
                                try:
                                    if self.piece_constrainer(self.position_to_string(position), self.position_to_string(move), piece=piece_key):
                                        return True
                                except Exception:
                                    continue
        return False
    def is_square_occupied_by_side(self, position, side):
        # Check if the square is occupied by a piece or pawn of the same side
        for piece, bitboard in self.board.items():
            if piece.startswith(side) and (bitboard & (1 << position)):
                return True
        return False
    def can_any_pawn_move(self, side):
        direction = -8 if side == 'w' else 8
        double_step_rank_mask = 0xFF0000 if side == 'w' else 0xFF0000000000
        pawn_bitboard = self.board[side + 'P']
        empty_squares = ~(self.board['wP'] | self.board['bP'] | self.board['wR'] | self.board['bR'] | self.board['wN'] | self.board['bN'] | self.board['wB'] | self.board['bB'] | self.board['wQ'] | self.board['bQ'] | self.board['wK'] | self.board['bK'])

        # Single step moves
        one_step_moves = (pawn_bitboard << direction) & empty_squares
        if one_step_moves:
            return True

        # Double step moves
        two_step_moves = ((one_step_moves & double_step_rank_mask) << direction) & empty_squares
        if two_step_moves:
            return True

        # Captures (including en passant)
        all_opponents = self.board['bP'] | self.board['bR'] | self.board['bN'] | self.board['bB'] | self.board['bQ'] | self.board['bK'] if side == 'w' else self.board['wP'] | self.board['wR'] | self.board['wN'] | self.board['wB'] | self.board['wQ'] | self.board['wK']
        file_mask = 0x7E7E7E7E7E7E7E7E  # Mask to prevent wraparound captures
        left_captures = ((pawn_bitboard & file_mask) << (direction - 1)) & all_opponents
        right_captures = ((pawn_bitboard & file_mask) << (direction + 1)) & all_opponents
        if left_captures or right_captures:
            return True

        # En passant
        last_move = self.move_log[-1] if self.move_log else None
        if last_move:
            last_piece = self.get_piece_at_position(last_move[1])
            last_from_position = last_move[1]
            last_to_position = last_move[2]
            if last_piece and last_piece[1].lower() == 'p' and abs(last_from_position - last_to_position) == 16:
                # Calculate en passant target square
                en_passant_target = last_to_position + (direction if side == 'b' else -direction)
                if (pawn_bitboard & (1 << (en_passant_target - 1))) or (pawn_bitboard & (1 << (en_passant_target + 1))):
                    return True

        # No pawn can move
        return False
    
    def is_en_passant_move(self, from_position, to_position):
        # Check if the move is a diagonal pawn move
        if abs(from_position - to_position) == 7 or abs(from_position - to_position) == 9:
            #print("Checking en passant move")
            # Get the piece at the from_position
            from_square = self.position_to_string(from_position)
            moving_piece = self.get_piece_at_square(from_square)

            # Check if the piece is a pawn
            if moving_piece[1].lower() == 'p':
                # Get the last move from the move log
                last_move = self.move_log[-1]

                # Get the piece, from_position, and to_position of the last move
                last_piece = list(self.piece_enum.keys())[list(self.piece_enum.values()).index(last_move[0])]
                #print(f"Last move: {last_piece}, {last_move[1]}, {last_move[2]}")
                last_from_position = last_move[1]
                last_to_position = last_move[2]
                #print(last_piece, last_from_position, last_to_position)

                # Check if the last move was a two-step pawn move
                if last_piece[1].lower() == 'p' and abs(last_from_position - last_to_position) == 16:
                    #print("Two step pawn move detected")
                    # Check if the to_position is the en passant target
                    if to_position == last_from_position + 8 or to_position == last_from_position - 8:
                        return True

        return False
    
    def apply_pawn_constraints(self, from_position, to_position, pawn_type="w"):
        """
        Apply constraints to the movement of a pawn on the chess board.

        Args:
            self: The object itself.
            from_position: The current position of the pawn.
            to_position: The target position of the pawn.
            pawn_type: The type of pawn, 'w' for white and 'b' for black. Defaults to 'w'.

        Returns:
            bool: True if the move satisfies all constraints.
        
        Raises:
            ValueError: If the move violates any constraint.
        """
        from_rank, to_rank = from_position // 8, to_position // 8
        from_file, to_file = from_position % 8, to_position % 8
        from_square, to_square = self.position_to_string(from_position), self.position_to_string(to_position)

        # White pawns can only move to higher ranks (up the board), black pawns to lower ranks (down the board)
        if (pawn_type == "w" and to_rank >= from_rank) or (pawn_type == "b" and to_rank <= from_rank):
            
            raise ValueError(f"Pawn cannot move backwards: {from_square} to {to_square}")

        move_distance = abs(to_rank - from_rank)
        # Check if the pawn is on its second rank
        is_second_rank = (from_rank == 6 and pawn_type == "w") or (from_rank == 1 and pawn_type == "b")

        # Pawns can only move one square forward, or two squares forward from the second rank
        if not (move_distance == 1 or (is_second_rank and move_distance == 2)):
            
            raise ValueError(f"Invalid move for pawn: {from_square} to {to_square}")

        # Pawns can only move straight forward if not capturing
        if from_file == to_file:
            
            # The target square must be empty
            target_piece = self.get_piece_at_square(to_square)
            if target_piece:
                raise ValueError(f"Cannot move to occupied square: {to_square}")
            # Check for pawn promotion
            to_rank = to_position // 8
            if (pawn_type == "w" and to_rank == 0) or (pawn_type == "b" and to_rank == 7):
                # Prompt the user for the type of piece to promote to
                print("Pawn promotion!")
        else:
            # Check for en passant
            if self.is_en_passant_move(from_position, to_position):
                # Perform en passant capture
                captured_pawn_position = to_position - 8 if pawn_type == "b" else to_position + 8
                self.en_passant_target = captured_pawn_position
                captured_pawn_square = self.position_to_string(captured_pawn_position)
                captured_pawn = self.get_piece_at_square(captured_pawn_square)
                if captured_pawn:
                    self.clear_piece(captured_pawn, captured_pawn_position)
            else:
                # Pawns can only capture diagonally to the adjacent file of the successive rank
                if abs(to_file - from_file) != 1 or (pawn_type == "w" and to_rank != from_rank - 1) or (pawn_type == "b" and to_rank != from_rank + 1):
                    raise ValueError(f"Invalid capture move for pawn: {from_square} to {to_square}")

                # The target square must contain a piece of the opposite color
                target_piece = self.get_piece_at_square(to_square)
                if not target_piece or target_piece.startswith(pawn_type):
                    raise ValueError(f"No piece to capture or own piece at target square: {to_square}")

                # clear the target square if target_piece:
                self.clear_piece(target_piece, to_position)

        return True
    def get_bishop_obstacles(self, from_position, to_position):
        #print(f"Getting bishop obstacles from {self.position_to_string(from_position)} to {self.position_to_string(to_position)}")
        # Calculate the direction of the diagonal
        rank_diff = (to_position // 8) - (from_position // 8)
        #print(rank_diff)
        file_diff = (to_position % 8) - (from_position % 8)
        #print(file_diff)
        
        # Check if the move is diagonal
        if abs(rank_diff) != abs(file_diff):
            # Convert positions to square strings
            from_square = self.position_to_string(from_position)
            to_square = self.position_to_string(to_position)
            raise(InvalidBishopMoveError(f"Invalid move for bishop: {from_square} to {to_square}, not a diagonal move"))
        
        # Get the bitboard for all possible moves for the bishop from the lookup table
        possible_moves_bitboard = self.B_lookup[from_position]
        
        # Determine the direction of the move
        rank_direction = 1 if rank_diff > 0 else -1
        file_direction = 1 if file_diff > 0 else -1
        
        # Calculate the bitboard for the path from the from_position to to_position
        path_bitboard = 0
        current_position = from_position
        while current_position != to_position:
            current_position += rank_direction * 8 + file_direction
            if current_position == to_position:
                break
            path_bitboard |= 1 << current_position
        
        # Intersect the path bitboard with the possible moves bitboard to get the path without obstacles
        path_without_obstacles = path_bitboard & possible_moves_bitboard
        
        # Check for obstacles by intersecting the path without obstacles with the actual board state
        for piece, bitboard in self.board.items():
            if bitboard & path_without_obstacles:
                return True  # Obstacle found, return True immediately
        
        return False  # No obstacles found
    def apply_bishop_constraints(self, from_position, to_position):
        """
        Apply bishop constraints to the move from one position to another.

        Args:
            from_position: The position from which the move starts.
            to_position: The position to which the move ends.

        Returns:
            bool: True if the move satisfies the bishop constraints, otherwise False.
        """
        # Check if the path is clear
        obstacles = self.get_bishop_obstacles(from_position, to_position)

        # Check if there are obstacles in the path
        if obstacles:
            from_square = self.position_to_string(from_position)
            to_square = self.position_to_string(to_position)
            raise ValueError(f"Invalid move for bishop: path from {from_square} to {to_square} is obstructed")

        return True
    
    def apply_knight_constraints(self, from_position, to_position):   
        """
        Apply constraints for the knight's movement on the chessboard.

        Parameters:
        - from_position: int, the starting position of the knight
        - to_position: int, the target position for the knight

        Returns:
        - bool: True if the move is valid, otherwise raises a ValueError
        """
        valid_moves = self.N_lookup[from_position]

        if valid_moves & (1 << to_position):
            return True
        else:
            from_square = self.position_to_string(from_position)
            to_square = self.position_to_string(to_position)
            self.illegal_played = True
            raise ValueError(f"Invalid move for knight: {from_square} to {to_square} is not a legal knight move")

    def get_rook_obstacles(self, from_position, to_position):
        rank_diff = (to_position // 8) - (from_position // 8)
        file_diff = (to_position % 8) - (from_position % 8)
        
        if rank_diff != 0 and file_diff != 0:
            from_square, to_square = self.position_to_string(from_position), self.position_to_string(to_position)
            raise InvalidRookMoveError(f"Invalid move for rook : {from_square} to {to_square}, not a straight move")

        # Get the bitboard for all possible moves for the rook from the lookup table
        possible_moves_bitboard = self.R_lookup[from_position]
        
        # Calculate the bitboard for the path from the from_position to to_position
        path_bitboard = 0
        current_position = from_position
        if rank_diff == 0:
            # Move along the file
            file_direction = 1 if file_diff > 0 else -1
            while current_position != to_position:
                current_position += file_direction
                if current_position == to_position:
                    break
                path_bitboard |= 1 << current_position
        else:
            # Move along the rank
            rank_direction = 1 if rank_diff > 0 else -1
            while current_position != to_position:
                current_position += rank_direction * 8
                if current_position == to_position:
                    break
                path_bitboard |= 1 << current_position
        
        # Intersect the path bitboard with the possible moves bitboard to get the path without obstacles
        path_without_obstacles = path_bitboard & possible_moves_bitboard
        
        # Check for obstacles by intersecting the path without obstacles with the actual board state
        for piece, bitboard in self.board.items():
            if bitboard & path_without_obstacles:
                return True  # Obstacle found
        
        return False  # No obstacles found
    
    def apply_rook_constraints(self, from_position, to_position):
        """
        Apply rook constraints and check for obstacles in the path between two positions.
        Parameters:
            from_position: tuple, the starting position
            to_position: tuple, the destination position
        Returns:
            bool, True if the path is clear, False otherwise
        """
        # Check if the path is clear
        obstacles = self.get_rook_obstacles(from_position, to_position)

        # Check if there are obstacles in the path
        if obstacles:
            from_square, to_square = self.position_to_string(from_position), self.position_to_string(to_position)
            raise ValueError(f"Invalid move for rook: path from {from_square} to {to_square} is obstructed")

        return True
    def get_queen_obstacles(self, from_position, to_position):
        rank_diff = (to_position // 8) - (from_position // 8)
        file_diff = (to_position % 8) - (from_position % 8)

        # Validate the move for queen: must be along the same rank, file, or one of the diagonals it lies on
        if rank_diff != 0 and file_diff != 0 and abs(rank_diff) != abs(file_diff):
            from_square, to_square = self.position_to_string(from_position), self.position_to_string(to_position)
            raise InvalidQueenMoveError(f"Invalid move for queen from {from_square} to {to_square}, not a straight or diagonal move")

        # Get the bitboard for all possible moves for the queen from the lookup table
        possible_moves_bitboard = self.Q_lookup[from_position]

        # Calculate the bitboard for the path from the from_position to to_position
        path_bitboard = 0
        current_position = from_position
        if rank_diff != 0 and file_diff != 0:
            # Diagonal movement
            rank_direction = 1 if rank_diff > 0 else -1
            file_direction = 1 if file_diff > 0 else -1
            while current_position != to_position:
                current_position += rank_direction * 8 + file_direction
                if current_position == to_position:
                    break
                path_bitboard |= 1 << current_position
        else:
            # Straight movement
            if rank_diff == 0:
                # Move along the file
                file_direction = 1 if file_diff > 0 else -1
                while current_position != to_position:
                    current_position += file_direction
                    if current_position == to_position:
                        break
                    path_bitboard |= 1 << current_position
            else:
                # Move along the rank
                rank_direction = 1 if rank_diff > 0 else -1
                while current_position != to_position:
                    current_position += rank_direction * 8
                    if current_position == to_position:
                        break
                    path_bitboard |= 1 << current_position

        # Intersect the path bitboard with the possible moves bitboard to get the path without obstacles
        path_without_obstacles = path_bitboard & possible_moves_bitboard

        # Check for obstacles by intersecting the path without obstacles with the actual board state
        for piece, bitboard in self.board.items():
            if bitboard & path_without_obstacles:
                return True  # Obstacle found

        return False  # No obstacles found

    def apply_queen_constraints(self, from_position, to_position):
        """
        Apply queen constraints by checking for obstacles in the path from one position to another.

        Args:
            from_position: The starting position of the queen
            to_position: The destination position of the queen

        Returns:
            True if the path is clear, otherwise raises a ValueError
        """
        # Check if the path is clear
        obstacles = self.get_queen_obstacles(from_position, to_position)

        # Check if there are obstacles in the path
        if obstacles:
            from_square, to_square = self.position_to_string(from_position), self.position_to_string(to_position)
            raise ValueError(f"Invalid move for queen: path from {from_square} to {to_square} is obstructed")

        return True
    
    def get_king_position(self, side="w"):
        for piece, bitboard in self.board.items():
            if piece[0] == side and piece[1] == "K":
                position = 0
                while bitboard:
                    if bitboard & 1:
                        return position
                    bitboard >>= 1
                    position += 1

    def checking_queen_or_rook(self, king_position, side="w"):
        """
        Determine if there is a rook or queen checking the king, given the king's position and the side to check.
        
        :param king_position: The position of the king on the board.
        :param side: The side to check for rook or queen (default is "w" for white).
        :return: The position of the checking piece if found, otherwise None.
        """
        rank, file = divmod(king_position, 8)
        # Determine the opposite side
        opposite_side = 'b' if side == 'w' else 'w'
        rook_queen_pieces = [opposite_side + 'R', opposite_side + 'Q']
        # Check along the file for a rook or queen
        for direction in [-1, 1]:  # Check in both directions along the file
            current_file = file
            while True:
                current_file += direction
                if 0 <= current_file < 8:
                    position = rank * 8 + current_file
                    piece = self.get_piece_at_position(position=position)
                    if piece:
                        if piece in rook_queen_pieces and not self.get_rook_obstacles(from_position=position, to_position=king_position):
                            return position  # Return the position of the checking piece
                        break  # Stop if any piece is encountered
                else:
                    break  # Stop if edge of board is reached

        # Check along the rank for a rook or queen
        for direction in [-8, 8]:  # Check in both directions along the rank
            current_rank = rank
            while True:
                current_rank += direction // 8
                if 0 <= current_rank < 8:
                    position = current_rank * 8 + file
                    piece = self.get_piece_at_position(position)
                    if piece:
                        if piece in rook_queen_pieces and not self.get_rook_obstacles(from_position=position, to_position=king_position):
                            return position  # Return the position of the checking piece
                        break  # Stop if any piece is encountered
                else:
                    break  # Stop if edge of board is reached

        return None  # Return None if no checking piece is found
        
    def checking_bishop_or_queen(self, king_position, side="w"):
        """
        Determine if the opponent has a bishop or queen that can attack the king at the given position.
        
        :param king_position: int, the position of the king on the board
        :param side: str, optional, the side for which to check (default is "w" for white)
        :return: int or None, the position of the attacking bishop or queen, or None if no such piece exists
        """
        opposite_side = 'b' if side == 'w' else 'w'
        bishop_queen_pieces = [opposite_side + 'B', opposite_side + 'Q']
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # Up-left, up-right, down-left, down-right

        for direction in directions:
            current_position = king_position
            while True:
                rank, file = divmod(current_position, 8)
                new_rank = rank + direction[0]
                new_file = file + direction[1]

                # Check if the new position is within board bounds
                if 0 <= new_rank < 8 and 0 <= new_file < 8:
                    new_position = new_rank * 8 + new_file
                    piece = self.get_piece_at_position(new_position)

                    # If a piece is encountered, check if it's a bishop or queen
                    if piece in bishop_queen_pieces:
                        return new_position
                    elif piece:  # Any other piece blocks the check
                        break
                else:  # Reached the edge of the board
                    break

                current_position = new_position

        return None  # No bishop or queen checking the king
    
    def checking_knight(self, king_position, side="w"):
        """
        A function to check if the opponent's knight is checking the king.
        
        Parameters:
            self (obj): The object itself.
            king_position (int): The position of the king on the board.
            side (str): The side to check for the opponent's knight (default is "w" for white).
        
        Returns:
            int or None: The position of the checking knight if found, otherwise None.
        """
        # Determine the opposite side
        opposite_side = 'b' if side == 'w' else 'w'
        knight_moves = self.N_lookup[king_position]
        opposite_knight = opposite_side + 'N'
        for move in range(64):
            if knight_moves & (1 << move) and self.get_piece_at_position(move) == opposite_knight:
                return move
        # The king is not checked
        return None
    
    def checking_pawn(self, king_position, side="w"):
        """
        Determine if there is a pawn checking the king.

        Parameters:
            self (object): The object itself.
            king_position (int): The position of the king on the board.
            side (str): The side to check for pawns (default is "w").

        Returns:
            int or None: The position of the checking pawn, or None if the king is not checked.
        """
        # Determine the opposite side
        opposite_side = 'b' if side == 'w' else 'w'
        # Check the diagonals for a pawn
        pawn_moves = [9, 7] if opposite_side == 'w' else [-9, -7]
        for move in pawn_moves:
            check_position = king_position + move
            check_rank, check_file = divmod(check_position, 8)
            if 0 <= check_rank < 8 and 0 <= check_file < 8:
                if self.get_piece_at_position(check_position) == opposite_side + 'P':
                    return check_position
        # The king is not checked
        return None
        
    def determine_if_checked(self, side="w"):
        """
        Determine if the specified side's king is checked. 
        Args:
            self: The object itself.
            side (str): The side of the board to check for king's position, defaults to "w".
        Returns:
            bool: True if the king is checked, False otherwise.
        """
        #print(f"\nDetermining if {side}'s king is in check...")
        king_position = self.get_king_position(side)
        #print(f"{side}'s king is at {self.position_to_string(king_position)}")
        
        # Check each type of threat
        adj_k = self.are_kings_adjacent(side, king_position)
        if adj_k:
            #print(f"{side}'s king is in check from adjacent king")
            return adj_k
            
        QR_check = self.checking_queen_or_rook(king_position, side)
        if QR_check:
            #print(f"{side}'s king is in check from queen/rook at {self.position_to_string(QR_check)}")
            return QR_check
            
        BQ_check = self.checking_bishop_or_queen(king_position, side)
        if BQ_check:
            #print(f"{side}'s king is in check from bishop/queen at {self.position_to_string(BQ_check)}")
            return BQ_check
            
        N_check = self.checking_knight(king_position, side)
        if N_check:
            #print(f"{side}'s king is in check from knight at {self.position_to_string(N_check)}")
            return N_check
            
        P_check = self.checking_pawn(king_position, side)
        if P_check:
            #print(f"{side}'s king is in check from pawn at {self.position_to_string(P_check)}")
            return P_check
            
        #print(f"{side}'s king is not in check")
        return False
    
    def check_resolved(self):
        """
        Check if the current player is in check and update the check status for both white and black players accordingly.
        """
        # Update both sides' check status
        white_check = self.determine_if_checked(side="w")
        black_check = self.determine_if_checked(side="b")
        
        self.white_in_check = bool(white_check)
        self.black_in_check = bool(black_check)
        
        #print(f"Updated check status - White in check: {self.white_in_check}, Black in check: {self.black_in_check}")
    
    def check_if_rook_moved(self, side, rook_type):
        # Define the initial positions for the rooks
        initial_positions = {
            'w': {'k': 1 << 63, 'q': 1 << 56},
            'b': {'k': 1, 'q': 1 << 7}
        }
        
        # Get the current bitboard for the specified rook
        rook_key = side + 'R'
        current_rook_bitboard = self.board[rook_key]
        
        # Get the initial position for the specified rook
        initial_rook_position = initial_positions[side][rook_type]
        
        # Check if the rook has moved from its initial position
        has_moved = not (current_rook_bitboard & initial_rook_position)
        
        return has_moved
    
    def are_kings_adjacent(self, side, to_position):
        """
        Check if the king is moving to a square adjacent to the opponent's king.

        Args:
            side (str): The side moving, either 'w' or 'b'.
            to_position (int): The bitboard position to move the king to.

        Returns: 
            bool: True if the kings are adjacent after moving, False otherwise.
        """
        # Get opponent's king position
        opponent_king_pos = self.get_king_position('b' if side == 'w' else 'w')
        
        # Get adjacent squares to opponent's king
        adj_squares = self.get_adjacent_positions(opponent_king_pos)
        
        # Check if target position is adjacent 
        return to_position in adj_squares

    def apply_king_constraints(self, from_position, to_position):
        """
        Apply king constraints to the given positions.

        Args:
            from_position: The starting position of the king.
            to_position: The destination position of the king.

        Returns:
            bool: True if the move satisfies the king's constraints, False otherwise.
        
        Raises:
            ValueError: If the move does not satisfy the king's constraints, with a message indicating the invalid move.
        """
        rank_diff = abs((to_position // 8) - (from_position // 8))
        file_diff = abs((to_position % 8) - (from_position % 8))
        
        if self.are_kings_adjacent('w' if self.white_to_move else 'b', to_position):
            from_square, to_square = self.position_to_string(from_position), self.position_to_string(to_position)
            raise ValueError(f"Invalid move for king from {from_square} to {to_square}, adjacent to opponent's king")

        # King can move one step in any direction, so the rank and file difference should be at most 1
        if rank_diff <= 1 and file_diff <= 1 and not (rank_diff == 0 and file_diff == 0):
            return True
        
        else:
            from_square, to_square = self.position_to_string(from_position), self.position_to_string(to_position)
            raise ValueError(f"Invalid move for king from {from_square} to {to_square}, not a single step")
        
        
            

    def assign_checking_and_final_states(self):
        """
        Determines the checking and final states of the game based on the current position and player to move. 
        Updates the game state variables such as white_in_check, black_in_check, checkmated, is_drawn, etc. 
        Also handles user input for resetting the game in case of checkmate, stalemate, threefold repetition, and insufficient material. 
        """
        #print("\n=== Checking Game State ===")
        #print(f"Current turn: {'White' if self.white_to_move else 'Black'} to move")
        
        # Always update both sides' check status
        white_check_pos = self.determine_if_checked(side="w")
        black_check_pos = self.determine_if_checked(side="b")
        self.white_in_check = bool(white_check_pos)
        self.black_in_check = bool(black_check_pos)
        
        #print(f"White in check: {self.white_in_check} {'from ' + self.position_to_string(white_check_pos) if white_check_pos else ''}")
        #print(f"Black in check: {self.black_in_check} {'from ' + self.position_to_string(black_check_pos) if black_check_pos else ''}")
        
        # Check for checkmate - check the side whose turn it is
        if self.white_to_move and self.white_in_check:  # It's White's turn and they're in check
            print("Checking if White is checkmated")
            if self.side_is_checkmated(side="w"):
                self.checkmated = True
                print("White has been checkmated.")
                
        elif not self.white_to_move and self.black_in_check:  # It's Black's turn and they're in check
            print("Checking if Black is checkmated")
            if self.side_is_checkmated(side="b"):
                self.checkmated = True
                print("Black has been checkmated.")
                
        # Assess stalemate conditions - check the side that's about to move
        if self.white_to_move and not self.white_in_check:
            if self.side_is_stalemated(side="w"):
                self.is_drawn = True
                print("The game has ended in a draw due to stalemate")
                
        elif not self.white_to_move and not self.black_in_check:
            if self.side_is_stalemated(side="b"):
                self.is_drawn = True
                print("The game has ended in a draw due to stalemate.")
                
        if self.is_threefold_repetition() and not self.editing:
            self.is_drawn = True
            print("Draw by threefold repetition")
            
        if self.is_material_insufficient() and not self.editing:
            self.is_drawn = True
            print("Draw by insufficient material")
        #print("=== End Game State Check ===\n")
    
    def determine_if_checked_while_castling(self, king_position, side="w"):
        """
        Determine if the king is checked during castling.

        Parameters:
            king_position (tuple): The position of the king on the board.
            side (str, optional): The side of the board to check for castling (default is "w" for white).

        Returns:
            bool or tuple: Returns the position of the checking piece or False if the king is not checked.
        """
         
        QR_check = self.checking_queen_or_rook(king_position, side)
        BQ_check = self.checking_bishop_or_queen(king_position, side)
        N_check = self.checking_knight(king_position, side)
        P_check = self.checking_pawn(king_position, side) 
        if QR_check:
            return QR_check
        elif BQ_check:
            return BQ_check
        elif N_check:
            return N_check
        elif P_check:
            return P_check
        # The king is not checked
        return False
    
    def is_checked_while_castling(self, side="w", direction="k"):
      castle_dict = {
            "wk": ["f1", "g1"],
            "wq": ["d1", "c1"],
            "bk": ["f8", "g8"],
            "bq": ["d8", "c8"]
        }
      for square in castle_dict[side + direction]:
          if self.determine_if_checked_while_castling(king_position=self.string_to_position(square), side=side):
              return True
      return False
    
    def is_obstructed_while_castling(self, side="w", direction="k"):
      castle_dict = {
            "wk": ["f1", "g1"],
            "wq": ["d1", "c1"],
            "bk": ["f8", "g8"],
            "bq": ["d8", "c8"]
        }
      for square in castle_dict[side + direction]:
          if self.get_piece_at_square(square):
              return True
      return False
  
    def apply_castling_constraints(self, side="w", direction="k"):
        """
        Apply castling constraints based on the given side and direction.

        Parameters:
            side (str): The side for castling, either "w" for white or "b" for black. Defaults to "w".
            direction (str): The direction for castling, either "k" for king-side or "q" for queen-side. Defaults to "k".

        Returns:
            bool: True if the castling constraints are satisfied, False otherwise.
        """
        #print(f"\n=== Debugging Castling Constraints for {side} {direction}-side ===")
        castle_string_expanded = side + "_castle_" + direction
        castle_status = getattr(self, castle_string_expanded)
        #print(f"Castle status from attribute {castle_string_expanded}: {castle_status}")
        
        if not castle_status:
            #print(f"Castling rejected: {castle_string_expanded} is False")
            return False
            
        if side == "w" and self.white_in_check:
            #print("Castling rejected: White is in check")
            return False
            
        if side == "b" and self.black_in_check:
            #print("Castling rejected: Black is in check")
            return False
            
        if self.is_obstructed_while_castling(side=side, direction=direction):
            #print(f"Castling rejected: Path is obstructed for {side} {direction}-side")
            return False
            
        if self.is_checked_while_castling(side=side, direction=direction):
            #print(f"Castling rejected: King would pass through check for {side} {direction}-side")
            return False
            
        #print(f"All castling constraints passed for {side} {direction}-side!")
        return True

    def is_obstructed_while_castling(self, side="w", direction="k"):
      castle_dict = {
            "wk": ["f1", "g1"],
            "wq": ["d1", "c1"],
            "bk": ["f8", "g8"],
            "bq": ["d8", "c8"]
        }
      path_squares = castle_dict[side + direction]
      #print(f"\nChecking castling path obstruction for {side} {direction}-side")
      #print(f"Checking squares: {path_squares}")
      for square in path_squares:
          piece = self.get_piece_at_square(square)
          if piece:
              print(f"Found obstruction at {square}: {piece}")
              return True
      #print("No obstructions found in castling path")
      return False

    def king_or_rook_moved(self, moving_piece="", from_square=""):
            #print(f"\n=== Checking piece movement for castling rights ===")
            #print(f"Moving piece: {moving_piece}, from square: {from_square}")
            # Check if any kings have moved or castled
            if 'K' in moving_piece or "OO" in moving_piece or "OOO" in moving_piece:
                if self.white_to_move and (self.w_castle_k or self.w_castle_q):
                    #print("White king has moved or castled - removing castling rights")
                    self.w_castle_k = False
                    self.w_castle_q = False
                if not self.white_to_move and (self.b_castle_k or self.b_castle_q):
                    #print("Black king has moved or castled - removing castling rights")
                    self.b_castle_k = False
                    self.b_castle_q = False 
            # Check if any rooks have moved
            if self.check_if_rook_moved(side="w", rook_type="k"):
                #print("White kingside rook has moved")
                self.KR_moved_white = True
                self.w_castle_k = False
            if self.check_if_rook_moved(side="w", rook_type="q"):
                #print("White queenside rook has moved")
                self.QR_moved_white = True
                self.w_castle_q = False
            if self.check_if_rook_moved(side="b", rook_type="k"):
                #print("Black kingside rook has moved")
                self.KR_moved_black = True
                self.b_castle_k = False
            if self.check_if_rook_moved(side="b", rook_type="q"):
                #print("Black queenside rook has moved")
                self.QR_moved_black = True
                self.b_castle_q = False
            # check if kings have moved
            if "K" in moving_piece:
                if self.white_to_move and from_square == "e1":
                    #print("White king moved from e1 - removing castling rights")
                    self.K_moved_white = True
                    self.w_castle_k = False
                    self.w_castle_q = False
                if not self.white_to_move and from_square == "e8":
                    #print("Black king moved from e8 - removing castling rights")
                    self.K_moved_black = True
                    self.b_castle_k = False
                    self.b_castle_q = False
            #print(f"Current castling rights after move:")
            #print(f"White: Kingside={self.w_castle_k}, Queenside={self.w_castle_q}")
            #print(f"Black: Kingside={self.b_castle_k}, Queenside={self.b_castle_q}")
            #print("=== End castling rights check ===\n")

    def check_if_rook_moved(self, side, rook_type):
        # Define the initial positions for the rooks
        initial_positions = {
            'w': {'k': 1 << 63, 'q': 1 << 56},
            'b': {'k': 1, 'q': 1 << 7}
        }
        
        # Get the current bitboard for the specified rook
        rook_key = side + 'R'
        current_rook_bitboard = self.board[rook_key]
        
        # Get the initial position for the specified rook
        initial_rook_position = initial_positions[side][rook_type]
        
        # Check if the rook has moved from its initial position
        has_moved = not (current_rook_bitboard & initial_rook_position)
        
        #print(f"\nChecking if {side}'s {rook_type}-side rook has moved")
        #print(f"Current rook bitboard: {bin(current_rook_bitboard)}")
        #print(f"Initial position: {bin(initial_rook_position)}")
        #print(f"Has moved: {has_moved}")
        
        return has_moved
    
    def castling_move(self, side="w", direction="k"):
        """
        A function to perform a castling move in chess, based on the side and direction provided.
        Parameters:
        - side: a string indicating the side, either "w" for white or "b" for black (default is "w")
        - direction: a string indicating the direction, either "k" for kingside or "q" for queenside (default is "k")
        Raises:
        - ValueError if the relevant rook has been captured, or if the direction or side provided is invalid for castling move
        """
        # Check that the relevant rook has not been captured
        rook_presence = self.get_piece_at_square("a" + str(1 if side == "w" else 8)) or self.get_piece_at_square("h" + str(1 if side == "w" else 8))
        if not rook_presence or rook_presence != side + "R":
            raise ValueError("The relevant rook has been captured")
        if side == "w" and direction == "k" and self.white_to_move:
            # place the white king and rook
            self.clear_piece(piece=side + "K", position=self.string_to_position("e1"))
            self.set_piece(piece=side + "K", position=self.string_to_position("g1"))
            self.clear_piece(piece=side + "R", position=self.string_to_position("h1"))
            self.set_piece(piece=side + "R", position=self.string_to_position("f1"))
        elif side == "b" and direction == "k" and not self.white_to_move:
            # place the black king and rook
            self.clear_piece(piece=side + "K", position=self.string_to_position("e8"))
            self.set_piece(piece=side + "K", position=self.string_to_position("g8"))
            self.clear_piece(piece=side + "R", position=self.string_to_position("h8"))
            self.set_piece(piece=side + "R", position=self.string_to_position("f8"))
        elif side == "w" and direction == "q" and self.white_to_move:
            # place the white king and rook
            self.clear_piece(piece=side + "K", position=self.string_to_position("e1"))
            self.set_piece(piece=side + "K", position=self.string_to_position("c1"))
            self.clear_piece(piece=side + "R", position=self.string_to_position("a1"))
            self.set_piece(piece=side + "R", position=self.string_to_position("d1"))
        elif side == "b" and direction == "q" and not self.white_to_move:
            # place the black king and rook
            self.clear_piece(piece=side + "K", position=self.string_to_position("e8"))
            self.set_piece(piece=side + "K", position=self.string_to_position("c8"))
            self.clear_piece(piece=side + "R", position=self.string_to_position("a8"))
            self.set_piece(piece=side + "R", position=self.string_to_position("d8"))
        else:
            raise ValueError("Invalid direction or wrong side for castling move")
        
    def castling_logic(self, moving_piece):
        """
        A function to handle castling logic for the given moving piece.

        Parameters:
        moving_piece (str): The type of castling move to be executed.

        Returns:
        None
        """
        # castling logic
        if "OO" in moving_piece or "OOO" in moving_piece:
            castle_type = moving_piece
            castle_direction = self.castle_mapping[castle_type]
            if self.white_to_move:
                self.castling_move(side="w", direction=castle_direction)
                self.log_castle("w"+castle_type)
                self.white_to_move = not self.white_to_move
            else:
                self.castling_move(side="b", direction=castle_direction)
                self.log_castle("b"+castle_type)
                self.white_to_move = not self.white_to_move
            
        
    def log_castle(self, castle_string):
        #print(castle_string)
        self.move_log = np.vstack([self.move_log, self.castle_logmap[castle_string]])
        #print(self.move_log) 


    def determine_if_king_cant_move_here(self, new_king_position, side="w"):
        """
        Determine if the king cannot move to the given position by checking for threats from queen, rook, bishop, knight, and pawn. 

        :param new_king_position: The new position of the king on the board.
        :param side: The side of the player, default is "w" for white.
        :return: Returns False if the king is not threatened, otherwise returns the type of piece that is threatening the king.
        """
        # Save current board state
        curr_king_position = self.get_king_position(side)
        old_board = deepcopy(self.board)
        
        # Temporarily move the king
        self.clear_piece(piece=side+"K", position=curr_king_position)
        self.set_piece(piece=side+"K", position=new_king_position)
        
        # Perform checks
        adj_k = self.are_kings_adjacent(side=side, to_position=new_king_position)
        if adj_k:
            self.board = old_board
            return adj_k

        QR_check = self.checking_queen_or_rook(new_king_position, side)
        if QR_check:
            self.board = old_board
            return QR_check

        BQ_check = self.checking_bishop_or_queen(new_king_position, side)
        if BQ_check:
            self.board = old_board
            return BQ_check

        N_check = self.checking_knight(new_king_position, side)
        if N_check:
            self.board = old_board
            return N_check

        P_check = self.checking_pawn(new_king_position, side)
        if P_check:
            self.board = old_board
            return P_check

        # Restore the board to the previous state once, before returning
        self.board = old_board
        return False

    def get_adjacent_positions(self, king_position):
        """
        Returns a list of adjacent positions to the given king_position on the board.
        
        Parameters:
        - king_position (int): The position of the king on the board (0-63).
        
        Returns:
        - list: A list of adjacent positions to the king_position on the board.
        """
        adjacent_positions = []
        # Offsets for all adjacent squares (including diagonals)
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        king_rank, king_file = divmod(king_position, 8)
        
        for rank_offset, file_offset in offsets:
            new_rank = king_rank + rank_offset
            new_file = king_file + file_offset
            
            # Check if the new position is on the board
            if 0 <= new_rank < 8 and 0 <= new_file < 8:
                # Convert the rank and file back to a single position and append
                adjacent_positions.append(new_rank * 8 + new_file)
        
        return adjacent_positions
    
    def king_can_move(self, side="w"):
        """
        Checks if the king can move to any adjacent positions. Returns True if the king can move, and False otherwise.
        Parameters:
            self (obj): The Chessboard object
            side (str): The side of the king, default is "w" (white)
        Returns:
            bool: True if the king can move, False otherwise
        """
        #print(f"Checking if {side}'s king can move...")
        king_position = self.get_king_position(side)
        adj_positions = self.get_adjacent_positions(king_position)
        non_moveable_positions = []
        
        # Save current board state
        old_board = deepcopy(self.board)
        king_piece = side + 'K'
        
        for position in adj_positions:
            occupying_piece = self.get_piece_at_position(position)
            if occupying_piece is not None and occupying_piece[0] == side:
                #print(f"King can't move to {self.position_to_string(position)} - blocked by own piece {occupying_piece}")
                non_moveable_positions.append(position)
                continue
            
            # Try moving the king
            try:
                # Clear king from current position
                self.clear_piece(king_piece, king_position)
                # Set king at new position (temporarily capturing any enemy piece except king)
                if occupying_piece and not occupying_piece.endswith('K'):
                    self.clear_piece(occupying_piece, position)
                self.set_piece(king_piece, position)
                
                # Check if this position is under attack
                if self.determine_if_checked(side=side):
                    #print(f"King can't move to {self.position_to_string(position)} - square is under attack")
                    non_moveable_positions.append(position)
                else:
                    #print(f"King can move to {self.position_to_string(position)}")
                    pass
                
                # Restore board state
                self.board = deepcopy(old_board)
                
            except Exception as e:
                print(f"Error checking move to {self.position_to_string(position)}: {str(e)}")
                self.board = deepcopy(old_board)
                non_moveable_positions.append(position)

        # If all adjacent positions are non-moveable, the king cannot move
        if set(adj_positions) == set(non_moveable_positions):
            #print(f"{side}'s king has no legal moves")
            return False
        else:
            #print(f"{side}'s king has legal moves")
            return True
    def can_piece_block_or_capture(self, target_position, side="w"):
        #print(f"\nChecking if any {side} piece can move to {self.position_to_string(target_position)}")
        # Get the position of the side's own king to skip it
        own_king_position = self.get_king_position(side)
        
        # Check for any piece that can reach the target position
        for piece, bitboard in self.board.items():
            if piece.startswith(side) and not piece.endswith('K'):  # Skip the king
                for position in range(64):
                    if bitboard & (1 << position) and position != own_king_position:  # Skip the king's position
                        try:
                            from_square = self.position_to_string(position)
                            to_square = self.position_to_string(target_position)
                            
                            #print(f"Checking if {piece} at {from_square} can move to {to_square}")
                            
                            # For pawns, validate diagonal moves strictly
                            if piece.endswith('P'):
                                from_file = position % 8
                                to_file = target_position % 8
                                from_rank = position // 8
                                to_rank = target_position // 8
                                
                                # If files are different (diagonal move)
                                if from_file != to_file:
                                    # Must have a piece to capture
                                    target_piece = self.get_piece_at_position(target_position)
                                    if not target_piece:
                                        #print(f"Pawn at {from_square} cannot move diagonally to {to_square} without a capture")
                                        continue
                                    # Must be moving in correct direction
                                    if (side == 'w' and to_rank >= from_rank) or (side == 'b' and to_rank <= from_rank):
                                        #print(f"Pawn at {from_square} cannot move backwards to {to_square}")
                                        continue
                                # If files are same (straight move)
                                else:
                                    # Cannot capture straight ahead
                                    target_piece = self.get_piece_at_position(target_position)
                                    if target_piece:
                                        #print(f"Pawn at {from_square} cannot capture straight ahead at {to_square}")
                                        continue
                                    # Must be moving in correct direction
                                    if (side == 'w' and to_rank >= from_rank) or (side == 'b' and to_rank <= from_rank):
                                        #print(f"Pawn at {from_square} cannot move backwards to {to_square}")
                                        continue
                            
                            # Save current board state
                            old_board = deepcopy(self.board)
                            
                            try:
                                # Try the move
                                if self.piece_constrainer(from_square=from_square, to_square=to_square, piece=piece):
                                    # Actually make the move
                                    self.clear_piece(piece, position)
                                    self.set_piece(piece, target_position)
                                    
                                    # Check if this resolves the check
                                    if not self.determine_if_checked(side=side):
                                        # Restore board state
                                        self.board = old_board
                                        #print(f"Found {piece} at {from_square} can move to {to_square} to block/capture")
                                        return True

                            except Exception as e:
                                pass
                            finally:
                                # Always restore board state
                                self.board = old_board
                                
                        except Exception as e:  # Catch any exception in the outer try block
                            print(f"Error processing {piece} at {from_square}: {str(e)}")
                            continue
        #print(f"No {side} piece can move to {self.position_to_string(target_position)}")
        return False
    
    def can_capture_or_block(self, checking_piece_position, side="w"):
        """ Determine if any piece of the specified side can capture the checking piece.
        Parameters:
            checking_piece_position (int): The position of the checking piece.
            side (str): The side ('w' for white, 'b' for black) of the pieces.
        Returns:
            bool: True if any piece can capture the checking piece, False otherwise.
        """
        print(f"\nChecking if {side} can block or capture piece at {self.position_to_string(checking_piece_position)}")
        # get king position
        king_position = self.get_king_position(side=side)
        rank_diff = (king_position // 8) - (checking_piece_position // 8)
        file_diff = (king_position % 8) - (checking_piece_position % 8)

        # Determine the check path
        path_positions = []
        if rank_diff == 0:
            #print("Check is along a rank")
            step = 1 if file_diff > 0 else -1
            path_positions = [(king_position // 8) * 8 + file for file in range(checking_piece_position % 8 + step, king_position % 8, step)]
        elif file_diff == 0:
            #print("Check is along a file")
            step = 8 if rank_diff > 0 else -8
            path_positions = [rank * 8 + (king_position % 8) for rank in range(checking_piece_position // 8 + step // 8, king_position // 8, step // 8)]
        elif abs(rank_diff) == abs(file_diff):
            #print("Check is along a diagonal")
            # Start from the checking piece and move towards the king
            rank = checking_piece_position // 8
            file = checking_piece_position % 8
            king_rank = king_position // 8
            king_file = king_position % 8
            
            # Determine step directions
            rank_step = 1 if king_rank > rank else -1
            file_step = 1 if king_file > file else -1
            
            # Move along diagonal until we reach the square before the king
            rank += rank_step
            file += file_step
            while (rank != king_rank) and (file != king_file):
                path_positions.append(rank * 8 + file)
                rank += rank_step
                file += file_step
            
            #print(f"Diagonal path from {self.position_to_string(checking_piece_position)} to {self.position_to_string(king_position)}")
        else:
            #print("Check is from a knight - no blocking possible")
            # If the check is not along a rank, file, or diagonal (e.g. knight), only capturing is possible
            path_positions = []

        # Always include the checking piece's position as it can be captured
        path_positions.append(checking_piece_position)
        print(f"Checking positions that could block or capture: {[self.position_to_string(pos) for pos in path_positions]}")
        
        # Check if any piece can block or capture along the path
        for pos in path_positions:
            if self.can_piece_block_or_capture(pos, side=side):
                #print(f"Found a piece that can move to {self.position_to_string(pos)}")
                return True
        #print("No pieces can block or capture")
        return False
    
    def side_is_checkmated(self, side="w"):
        print(f"\nChecking if {side} is checkmated...")
        checking_piece_position = self.determine_if_checked(side=side)
        if checking_piece_position:
            #print(f"{side} is in check from piece at {self.position_to_string(checking_piece_position)}")
            print("Checking if king can move...")
            if not self.king_can_move(side=side):
                #print(f"{side}'s king cannot move")
                print("Checking if any piece can block or capture...")
                if self.can_capture_or_block(checking_piece_position, side=side):
                    #print(f"Found a piece that can block or capture - not checkmate")
                    return False
                else:
                    print(f"No piece can block or capture - CHECKMATE!")
                    return True
            else:
                #print(f"{side}'s king can move - not checkmate")
                return False
        #print(f"{side} is not in check - not checkmate")
        return False
    
    def side_is_stalemated(self, side="w"):
        if self.king_can_move(side=side):
            return False
        if self.can_any_piece_move(side=side):
            return False
        if self.can_any_pawn_move(side=side):
            return False
        return True
    
    def is_threefold_repetition(self):
        board_hash = self.hash_board_state()
        return self.board_states.get(board_hash, 0) >= 3
    
    def is_material_insufficient(self):
        # Count the number of each piece on the board
        piece_counts = {piece: bin(self.board[piece]).count('1') for piece in self.board}

        # Check for a knight and two kings
        if piece_counts['wN'] == 1 and piece_counts['bN'] == 0 and \
           piece_counts['wK'] == 1 and piece_counts['bK'] == 1 and \
           sum(piece_counts.values()) == 3:
            return True

        # Check for two knights and two kings
        if piece_counts['wN'] == 2 and piece_counts['bN'] == 0 and \
           piece_counts['wK'] == 1 and piece_counts['bK'] == 1 and \
           sum(piece_counts.values()) == 4:
            return True

        # Check for a bishop and two kings
        if piece_counts['wB'] == 1 and piece_counts['bB'] == 0 and \
           piece_counts['wK'] == 1 and piece_counts['bK'] == 1 and \
           sum(piece_counts.values()) == 3:
            return True

        # Check for the mirror cases with black pieces instead of white
        if piece_counts['bN'] == 1 and piece_counts['wN'] == 0 and \
           piece_counts['bK'] == 1 and piece_counts['wK'] == 1 and \
           sum(piece_counts.values()) == 3:
            return True

        if piece_counts['bN'] == 2 and piece_counts['wN'] == 0 and \
           piece_counts['bK'] == 1 and piece_counts['wK'] == 1 and \
           sum(piece_counts.values()) == 4:
            return True

        if piece_counts['bB'] == 1 and piece_counts['wB'] == 0 and \
           piece_counts['bK'] == 1 and piece_counts['wK'] == 1 and \
           sum(piece_counts.values()) == 3:
            return True

        # If none of the conditions are met, return False
        return False
    

    def piece_constrainer(self, from_square="", to_square="", piece="wP"):
        """
        Check if a given piece follows the specified constraints for a move.

        Parameters:
            from_square (str): The starting position of the piece.
            to_square (str): The ending position of the piece.
            piece (str): The type of piece being moved. Defaults to "wP".

        Returns:
            bool: True if the piece follows the constraints, False otherwise.
        """
        try:
            follows_constraint = False
            # First check if it's the right color's turn to move
            if piece and not "O" in piece:  # Skip this check for castling notation (OO/OOO)
                is_white_piece = piece.startswith('w')
                if (is_white_piece and not self.white_to_move) or (not is_white_piece and self.white_to_move):
                    raise ValueError(f"Not {piece[0]}'s turn to move")

            if self.white_to_move and piece[0] == "w" or not self.white_to_move and piece[0] == "b" or "O" in piece:
                # Check for castling first - if the king is moving two squares, it must be a castling attempt
                if piece and piece[1] == "K" and from_square and to_square:  # Only check for castling if we have both squares
                    from_pos = self.string_to_position(from_square)
                    to_pos = self.string_to_position(to_square)
                    # Check if this is a castling move
                    if abs(from_pos % 8 - to_pos % 8) == 2 and from_square[1] == to_square[1]:
                        direction = "k" if to_pos % 8 > from_pos % 8 else "q"
                        side = piece[0]
                        follows_constraint = self.apply_castling_constraints(side=side, direction=direction)
                        return follows_constraint
            
                if piece[1] == "P" or piece == None:
                    follows_constraint = self.apply_pawn_constraints(from_position=self.string_to_position(from_square), to_position=self.string_to_position(to_square), pawn_type=piece[0])
                
                elif piece[1] == "B":
                    follows_constraint = self.apply_bishop_constraints(from_position=self.string_to_position(from_square), to_position=self.string_to_position(to_square))

                elif piece[1] == "N":
                    follows_constraint = self.apply_knight_constraints(from_position=self.string_to_position(from_square), to_position=self.string_to_position(to_square))
                
                elif piece[1] == "R":
                    follows_constraint = self.apply_rook_constraints(from_position=self.string_to_position(from_square), to_position=self.string_to_position(to_square))
                elif piece[1] == "Q":
                    follows_constraint = self.apply_queen_constraints(from_position=self.string_to_position(from_square), to_position=self.string_to_position(to_square))
                
                elif piece[1] == "K":
                    follows_constraint = self.apply_king_constraints(from_position=self.string_to_position(from_square), to_position=self.string_to_position(to_square))
                elif piece == "OO": # kingside castling
                    if self.white_to_move:
                        follows_constraint = self.apply_castling_constraints(side="w", direction="k")
                    else:
                        follows_constraint = self.apply_castling_constraints(side="b", direction="k")
                elif piece == "OOO": # queenside castling
                    if self.white_to_move:
                        follows_constraint = self.apply_castling_constraints(side="w", direction="q")
                    else:
                        follows_constraint = self.apply_castling_constraints(side="b", direction="q")
                            
                else:
                    # log warning saying constraint isn't implemented
                    follows_constraint = True
                    warnings.warn(f"Constraint not implemented for {piece[1]}")
            else:
                raise ValueError(f"Not the right color to move: {piece}")

            return follows_constraint

        except Exception as e:
            raise e
    def piece_castle_conditional(self, piece):
        if piece is None:
            return True
        elif "O" in piece:
            return False
        elif "O" not in piece:
            return True
        
    def move_piece(self, from_square="", to_square="", piece=None):
        """
        Move a piece from one square to another.
        """
        # Reset illegal_played at the start of each move attempt
        self.illegal_played = False
        
        # Don't proceed if we don't have both squares for any move
        if from_square and not to_square:
            print("Waiting for destination square...")
            return

        old_self = ConstrainedGameState()
        old_board = self.board.copy()
        for attr, value in vars(self).items():
            if isinstance(value, (bool, list, np.ndarray)):
                setattr(old_self, attr, deepcopy(value))
                
        # Convert single letter piece to full piece name with color
        if piece is not None and len(piece) == 1:
            if self.white_to_move:
                piece = 'w' + piece
            else:
                piece = 'b' + piece
                
        # Early check for correct turn
        if piece and not "O" in piece:  # Skip this check for castling notation (OO/OOO)
            is_white_piece = piece.startswith('w')
            if (is_white_piece and not self.white_to_move) or (not is_white_piece and self.white_to_move):
                self.illegal_played = True
                raise ValueError(f"Not {piece[0]}'s turn to move")
                
        try:
            if self.piece_castle_conditional(piece):
                from_position = self.string_to_position(from_square)
                to_position = self.string_to_position(to_square)
                moving_piece = None
                for p, bitboard in self.board.items():
                    if bitboard & (1 << from_position):
                        if piece is None or p == piece:
                            moving_piece = p
                            break
                if moving_piece is None:
                    self.illegal_played = True
                    raise ValueError(f"No {piece} at {from_square}.")
                    
                promotion = False
                if 'P' in moving_piece and ((moving_piece.startswith('w') and to_position // 8 == 0) or (moving_piece.startswith('b') and to_position // 8 == 7)):
                    promotion = True
                    self.promoting = True
                else:
                    self.promoting = False
                    
                if self.piece_constrainer(from_square, to_square, piece=moving_piece):
                    # Handle castling move
                    if moving_piece and moving_piece[1] == 'K' and abs(from_position % 8 - to_position % 8) == 2:
                        direction = "k" if to_position % 8 > from_position % 8 else "q"
                        side = moving_piece[0]
                        self.castling_move(side=side, direction=direction)
                        self.log_castle(side + ("OO" if direction == "k" else "OOO"))
                    else:
                        # Handle regular move
                        self.clear_piece(moving_piece, from_position)
                        if not promotion:
                            self.set_piece(moving_piece, to_position)
                        if promotion:
                            self.promoting_from_pos = from_position
                            self.promoting_to_pos = to_position
                        
                    self.king_or_rook_moved(moving_piece, from_square=from_square)

                    
                    self.update_board_state()
                    
                    # Check if the move was legal by verifying check status
                    white_check = self.determine_if_checked(side="w")
                    black_check = self.determine_if_checked(side="b")
                    
                    # A move is illegal if it leaves or puts your own king in check
                    if self.white_to_move and white_check:
                        self.illegal_played = True
                        raise ValueError("Illegal move. White is still in check or is put in check.")
                    if not self.white_to_move and black_check:
                        self.illegal_played = True
                        raise ValueError("Illegal move. Black is still in check or is put in check.")
                    
                    # Update check status and evaluate checkmate
                    self.white_in_check = bool(white_check)
                    self.black_in_check = bool(black_check)
                    
                    # Toggle turn
                    self.white_to_move = not self.white_to_move
                    
                    # Now check for checkmate/stalemate with the new turn state
                    self.assign_checking_and_final_states()
                    
                    if not self.promoting:
                        move_array = np.array([self.piece_enum[moving_piece], from_position, to_position], dtype=np.int8)
                        self.move_log = np.vstack([self.move_log, move_array])
                    if self.first_move:
                        self.move_log = self.move_log[1:]
                        self.first_move = False
                else:
                    self.illegal_played = True
                    raise ValueError(f"Illegal move from {from_square} to {to_square} for {moving_piece}.")
            else:
                if self.piece_constrainer(piece=piece):
                    self.king_or_rook_moved(moving_piece=piece)
                    self.castling_logic(piece)
                    self.update_board_state()
                    
        except Exception as e:
            # Save illegal_played state
            was_illegal = self.illegal_played
            # Restore the previous state if the move was invalid
            for attr in vars(old_self):
                setattr(self, attr, getattr(old_self, attr))
            self.board = old_board
            # Restore illegal_played to its value after the error
            self.illegal_played = was_illegal
            print(f"An error occurred while moving the piece: {e}")
            traceback.print_exc(file=sys.stdout)
            return
    def play_move(self):
        move_prompt = input("Enter your move: ").strip()

        # Castling moves
        if move_prompt == "OO" or move_prompt == "OOO":
            self.move_piece(piece=move_prompt)
            return

        # Piece-specific moves (e.g., "Ng1f3")
        if len(move_prompt) == 5 and move_prompt[0].isalpha() and move_prompt[1].isalpha():
            piece = move_prompt[0]
            from_square = move_prompt[1:3]
            to_square = move_prompt[3:5]
            
            self.move_piece(from_square=from_square, to_square=to_square, piece=piece)
            return

        # Normal moves (e.g., "e2e4")
        if len(move_prompt) == 4:
            from_square = move_prompt[:2]
            to_square = move_prompt[2:]
            self.move_piece(from_square=from_square, to_square=to_square)
            print(self.get_piece_at_square("g1"))
            return

        print("Invalid move format.")

def gameplay_check():
    game_state = ConstrainedGameState()
    game_state.set_start_position()
    game_state.move_piece(from_square="e2", to_square="e4")
    game_state.move_piece(from_square="e7", to_square="e5")
    
    game_state.move_piece(from_square="g1", to_square="f3", piece="N")
    game_state.move_piece(from_square="b8", to_square="c6", piece="N")
    game_state.move_piece(from_square="f1", to_square="b5", piece="B")
    game_state.move_piece(from_square="a7", to_square="a6")
    game_state.move_piece(from_square="b5", to_square="a4", piece="B")
    game_state.move_piece(from_square="h7", to_square="h5")
    game_state.move_piece(from_square="a4", to_square="b3", piece="B")
    game_state.move_piece(from_square="h8", to_square="h6", piece="R")
    game_state.move_piece(from_square="d1", to_square="e2", piece="Q")
    game_state.move_piece(from_square="h5", to_square="h4")
    game_state.move_piece(piece="OO")
    
    game_state.display_board()
    move_dict = game_state.get_all_possible_moves_current_pos(side="w")
    print(move_dict.keys())
    
def checking_evaluation():
    game_state = ConstrainedGameState()
    game_state.set_start_position()
    game_state.clear_board()
    game_state.set_piece("wP", game_state.string_to_position("e7"))
    game_state.set_piece("bK", game_state.string_to_position("c2"))
    game_state.set_piece("wK", game_state.string_to_position("e1"))
    game_state.move_piece(from_square="e7", to_square="e8")
    game_state.display_board()

def checkmate_evaluation():
    game_state = ConstrainedGameState()
    game_state.set_start_position()
    game_state.move_piece(from_square="g2", to_square="g4")
    game_state.move_piece(from_square="e7", to_square="e5")
    game_state.move_piece(from_square="f2", to_square="f4")
    game_state.move_piece(from_square="d8", to_square="h4", piece="Q") # Should be checkmate
    game_state.display_board()

def threefold_rep_evaluation():
    game_state = ConstrainedGameState()
    game_state.set_piece("wP", game_state.string_to_position("e2")) # direct piece placement
    game_state.set_piece("bK", game_state.string_to_position("e5"))
    game_state.set_piece("wK", game_state.string_to_position("e1"))
    game_state.set_piece("bP", game_state.string_to_position("e4"))
    game_state.white_to_move = True
    game_state.display_board()
    game_state.move_piece(from_square="e1", to_square="f1", piece="K")
    game_state.move_piece(from_square="e5", to_square="f5", piece="K")
    game_state.move_piece(from_square="f1", to_square="e1", piece="K")
    game_state.move_piece(from_square="f5", to_square="e5", piece="K")
    
    game_state.display_board()
    
    game_state.move_piece(from_square="e1", to_square="f1", piece="K")
    game_state.move_piece(from_square="e5", to_square="f5", piece="K")
    game_state.move_piece(from_square="f1", to_square="e1", piece="K")
    game_state.move_piece(from_square="f5", to_square="e5", piece="K")
    
    game_state.display_board()
    
    game_state.move_piece(from_square="e1", to_square="f1", piece="K")
    game_state.move_piece(from_square="e5", to_square="f5", piece="K")
    game_state.move_piece(from_square="f1", to_square="e1", piece="K")
    game_state.move_piece(from_square="f5", to_square="e5", piece="K")
    
    game_state.display_board()
    
if __name__ == "__main__":
    #gameplay_check()
    checking_evaluation()
    #checkmate_evaluation()
    #threefold_rep_evaluation()



