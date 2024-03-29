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
    
    def move_piece(self, from_square, to_square, piece=None):
        # Save the current state of the board
        old_board = self.board.copy()
        old_move_log = self.move_log.copy()
        old_white_to_move = self.white_to_move

        try:
            # Convert the squares to positions
            from_position = self.string_to_position(from_square)
            to_position = self.string_to_position(to_square)

            # Determine the color and type of the piece at the from_position
            moving_piece = None
            for p, bitboard in self.board.items():
                if bitboard & (1 << from_position):
                    if piece is None or p[1] == piece:
                        moving_piece = p
                        break
            if moving_piece is None:
                raise ValueError(f"No {piece} at {from_square}.")

            # Clear the old position
            self.clear_piece(moving_piece, from_position)

            # Set the new position
            self.set_piece(moving_piece, to_position)

            # toggle self.white_to_move
            self.white_to_move = not self.white_to_move

            # log the move
            move_array = np.array([self.piece_enum[moving_piece], from_position, to_position], dtype=np.int8)
            self.move_log = np.vstack([self.move_log, move_array])
            if self.first_move:
                self.move_log = self.move_log[1:]
                self.first_move = False

        except Exception as e:
            # If an error occurred during the move, revert the board state to the previous state
            self.board = old_board
            self.move_log = old_move_log
            self.white_to_move = old_white_to_move
            print(f"An error occurred while moving the piece: {e}")
    
    def display_board(self):
        print(self.get_board_representation())
        
class ConstrainedGameState(GameState):
    def __init__(self):
        super().__init__()
        self.board_states = {}
        self.P_lookup_w = self.get_all_possible_pawn_moves_for_side('w')
        self.P_lookup_b = self.get_all_possible_pawn_moves_for_side('b')
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
    
    def get_all_possible_knight_moves(self):
        knight_moves = {}
        for i in range(64):
            possible_moves_bitboard = 0  
            if i - 17 >= 0 and i % 8 >= 1:
                possible_moves_bitboard |= 1 << (i - 17)
            if i - 15 >= 0 and i % 8 <= 6:
                possible_moves_bitboard |= 1 << (i - 15)
            if i + 17 < 64 and i % 8 <= 6:
                possible_moves_bitboard |= 1 << (i + 17)
            if i + 15 < 64 and i % 8 >= 1:
                possible_moves_bitboard |= 1 << (i + 15)
            if i - 10 >= 0 and i % 8 >= 2:
                possible_moves_bitboard |= 1 << (i - 10)
            if i - 6 >= 0 and i % 8 <= 5:
                possible_moves_bitboard |= 1 << (i - 6)
            if i + 10 < 64 and i % 8 <= 5:
                possible_moves_bitboard |= 1 << (i + 10)
            if i + 6 < 64 and i % 8 >= 2:
                possible_moves_bitboard |= 1 << (i + 6)
            knight_moves[i] = possible_moves_bitboard
        return knight_moves
    def get_all_possible_bishop_moves(self):
        bishop_moves = {}
        for i in range(64):
            possible_moves_bitboard = 0
            x, y = divmod(i, 8)
            for dx in range(-7, 8):
                if dx == 0:
                    continue
                new_x, new_y = x + dx, y + dx
                if 0 <= new_x < 8 and 0 <= new_y < 8:
                    possible_moves_bitboard |= 1 << (new_x * 8 + new_y)
                new_x, new_y = x + dx, y - dx
                if 0 <= new_x < 8 and 0 <= new_y < 8:
                    possible_moves_bitboard |= 1 << (new_x * 8 + new_y)
            bishop_moves[i] = possible_moves_bitboard
        return bishop_moves

    def get_all_possible_rook_moves(self):
        rook_moves = {}
        for i in range(64):
            possible_moves_bitboard = 0
            x, y = divmod(i, 8)
            for dx in range(-7, 8):
                if dx != 0:
                    new_x, new_y = x + dx, y
                    if 0 <= new_x < 8:
                        possible_moves_bitboard |= 1 << (new_x * 8 + new_y)
                    new_x, new_y = x, y + dx
                    if 0 <= new_y < 8:
                        possible_moves_bitboard |= 1 << (new_x * 8 + new_y)
            rook_moves[i] = possible_moves_bitboard
        return rook_moves
    def get_all_possible_pawn_moves(self):
        pawn_moves_w = self.get_all_possible_pawn_moves_for_side('w')
        pawn_moves_b = self.get_all_possible_pawn_moves_for_side('b')
        return {**pawn_moves_w, **pawn_moves_b}
    
    def get_all_possible_pawn_moves_for_side(self, side):
        direction = -8 if side == 'w' else 8
        promotion_rank_mask = 0xFF00000000000000 if side == 'w' else 0xFF
        double_step_rank_mask = 0xFF00 if side == 'w' else 0xFF000000000000
        pawn_moves = {}
        
        pawn_bitboard = self.board[side + 'P']
        empty_squares = ~(self.board['wP'] | self.board['bP'] | self.board['wR'] | self.board['bR'] | self.board['wN'] | self.board['bN'] | self.board['wB'] | self.board['bB'] | self.board['wQ'] | self.board['bQ'] | self.board['wK'] | self.board['bK'])

        for position in range(64):
            if not (pawn_bitboard & (1 << position)):
                continue  # Skip if there's no pawn at the current position
            
            possible_moves_bitboard = 0
            
            # Single step moves
            one_step_forward = (1 << (position + direction)) & empty_squares
            if one_step_forward:
                if one_step_forward & promotion_rank_mask:  # Check for promotion
                    # Generate promotion moves
                    for promotion_piece in ["=wQ", "=wR", "=wN", "=wB"] if side == 'w' else ["=bQ", "=bR", "=bN", "=bB"]:
                        promotion_move = self.piece_enum[promotion_piece]  # Use your piece_enum to encode the promotion
                        pawn_moves[position] = promotion_move  # Store the promotion move
                else:
                    possible_moves_bitboard |= one_step_forward
                    
                # Double step moves from 2nd rank
                if (position // 8 == 6 and side == 'w') or (position // 8 == 1 and side == 'b'):
                    two_step_forward = (1 << (position + 2 * direction)) & empty_squares & double_step_rank_mask
                    if two_step_forward:
                        possible_moves_bitboard |= two_step_forward
                    
            # Captures and En Passant logic remains the same
            
            if possible_moves_bitboard:
                pawn_moves[position] = possible_moves_bitboard
        
        return pawn_moves

    def get_all_possible_queen_moves(self):
        queen_moves = {}
        for i in range(64):
            possible_moves_bitboard = 0
            x, y = divmod(i, 8)
            # Diagonal moves (like bishop)
            for dx in range(-7, 8):
                if dx == 0:
                    continue
                new_x, new_y = x + dx, y + dx
                if 0 <= new_x < 8 and 0 <= new_y < 8:
                    possible_moves_bitboard |= 1 << (new_x * 8 + new_y)
                new_x, new_y = x + dx, y - dx
                if 0 <= new_x < 8 and 0 <= new_y < 8:
                    possible_moves_bitboard |= 1 << (new_x * 8 + new_y)
            # Straight moves (like rook)
            for dx in range(-7, 8):
                if dx != 0:
                    new_x, new_y = x + dx, y
                    if 0 <= new_x < 8:
                        possible_moves_bitboard |= 1 << (new_x * 8 + new_y)
                    new_x, new_y = x, y + dx
                    if 0 <= new_y < 8:
                        possible_moves_bitboard |= 1 << (new_x * 8 + new_y)
            queen_moves[i] = possible_moves_bitboard
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
                    piece = self.get_piece_at_position(position)
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
        ##print("Getting king position...")
        king_position = self.get_king_position(side)
        #print("King position:", self.position_to_string(king_position))
        #print("Checking if king is checked by rook or queen...")
        adj_k = self.are_kings_adjacent(side, king_position)
        QR_check = self.checking_queen_or_rook(king_position, side)
        #print("Checking if king is checked by bishop or queen...")
        BQ_check = self.checking_bishop_or_queen(king_position, side)
        #print("Checking if king is checked by knight...")
        N_check = self.checking_knight(king_position, side)
        #print("Checking if king is checked by pawn...")
        P_check = self.checking_pawn(king_position, side)
        #print("All checks done.") 
        if QR_check:
            return QR_check
        elif BQ_check:
            #print("BQ check")
            
            return BQ_check
        elif N_check:
            return N_check
        elif P_check:
            return P_check
        elif adj_k:
            return adj_k
        # The king is not checked
        return False
    
    def check_resolved(self):
        """
        Check if the current player is in check and update the check status for both white and black players accordingly.
        """
        side = "w" if  not self.white_to_move else "b"
        if self.determine_if_checked(side=side):
            if not self.white_to_move:
                self.white_in_check = True
            else:
                self.black_in_check = True
        else:
            if not self.white_to_move:
                self.white_in_check = False
            else:
                self.black_in_check = False
    
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
        #self.white_in_check = True if self.determine_if_checked(side="w") and self.white_to_move else False
        #self.black_in_check = True if self.determine_if_checked(side="b") and not self.white_to_move else False
        if self.white_to_move:
            if self.determine_if_checked(side="w"):
                self.white_in_check = True
            
        if not self.white_to_move:
            #print("Checking if black is in check...")
            if self.determine_if_checked(side="b"):
                #print("Black is in check.")
                self.black_in_check = True
        if self.white_to_move and self.white_in_check:
            if self.side_is_checkmated(side="w"):
                self.checkmated = True
                # Tell the user that white has been checkmated
                print("White has been checkmated.")
                
        elif not self.white_to_move and self.black_in_check:
            print("Checking if black is checkmated...")
            if self.side_is_checkmated(side="b"):
                self.checkmated = True
                # Tell the user that black has been checkmated
                print("Black has been checkmated.")
                
                
        # Assess stalemate conditions
        if self.white_to_move and not self.white_in_check:
            if self.side_is_stalemated(side="w"):
                self.is_drawn = True
                # Tell the user that the game has ended in a draw due to stalemate
                print("The game has ended in a draw due to stalemate")
                
        elif not self.white_to_move and not self.black_in_check:
            if self.side_is_stalemated(side="b"):
                self.is_drawn = True
                # Tell the user that the game has ended in a draw due to stalemate
                print("The game has ended in a draw due to stalemate.")
                
        if self.is_threefold_repetition() and not self.editing:
            self.is_drawn = True
            print("Draw by threefold repetition")
            
        if self.is_material_insufficient() and not self.editing:
            self.is_drawn = True
            print("Draw by insufficient material")
            
                           
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
        castle_string_expanded = side + "_castle_" + direction
        castle_status = getattr(self, castle_string_expanded)
        #print("I've got the castle status", castle_status)
        if not castle_status:
            return False
        if side == "w" and self.white_in_check:
            return False
        if side == "b" and self.black_in_check:
            return False
        if self.is_obstructed_while_castling(side=side, direction=direction):
            return False
        if self.is_checked_while_castling(side=side, direction=direction):
            return False
        return True
    
    def king_or_rook_moved(self, moving_piece="", from_square=""):
            # Check if any kings have moved or castled
            if 'K' in moving_piece or "OO" in moving_piece or "OOO" in moving_piece:
                if self.white_to_move and (self.w_castle_k or self.w_castle_q):
                    #print("White can't castle again")
                    self.w_castle_k = False
                    self.w_castle_q = False
                if not self.white_to_move and (self.b_castle_k or self.b_castle_q):
                    self.b_castle_k = False
                    self.b_castle_q = False 
            # Check if any rooks have moved
            if self.check_if_rook_moved(side="w", rook_type="k"):
                self.KR_moved_white = True
                self.w_castle_k = False
            if self.check_if_rook_moved(side="w", rook_type="q"):
                self.QR_moved_white = True
                self.w_castle_q = False
            if self.check_if_rook_moved(side="b", rook_type="k"):
                self.KR_moved_black = True
                self.b_castle_k = False
            if self.check_if_rook_moved(side="b", rook_type="q"):
                self.QR_moved_black = True
                self.b_castle_q = False
            # check if kings have moved
            if "K" in moving_piece:
                if self.white_to_move and from_square == "e1":
                    self.K_moved_white = True
                    self.w_castle_k = False
                    self.w_castle_q = False
                if not self.white_to_move and from_square == "e8":
                    self.K_moved_black = True
                    self.b_castle_k = False
                    self.b_castle_q = False
            
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
        king_position = self.get_king_position(side)
        adj_positions = self.get_adjacent_positions(king_position)
        non_moveable_positions = []
        for position in adj_positions:
            occupying_piece = self.get_piece_at_position(position)
            if occupying_piece is not None and occupying_piece[0] == side:
                #print(f"King can't move to {self.position_to_string(position)}, because of own piece")
                non_moveable_positions.append(position) # piece is present at position, therefore king can't move here
            elif self.determine_if_king_cant_move_here(position, side):
                #print(f"King can't move to {self.position_to_string(position)}, because of check")
                non_moveable_positions.append(position) # king is in check here, therefore king can't move here

        # If all adjacent positions are non-moveable, the king cannot move
        #adj_squares = [self.position_to_string(position) for position in adj_positions]
        #non_movable_squares = [self.position_to_string(position) for position in non_moveable_positions]
        #print(f"Adjacent squares: {adj_squares}\n Non-moveable squares: {non_movable_squares}")
        if set(adj_positions) == set(non_moveable_positions):
            return False
        else:
            return True
    def can_piece_block_or_capture(self, target_position, side="w"):
        # Get the position of the side's own king to skip it
        own_king_position = self.get_king_position(side)
        
        # Check for any piece that can reach the target position
        for piece, bitboard in self.board.items():
            if piece.startswith(side) and not piece.endswith('K'):  # Skip the king
                for position in range(64):
                    #print(f"Trying position {self.position_to_string(position)} for {piece}")
                    if bitboard & (1 << position) and position != own_king_position:  # Skip the king's position
                        #print("Present")
                        # Check if the piece can legally move to the target position
                        try:
                            if self.piece_constrainer(from_square=self.position_to_string(position), to_square=self.position_to_string(target_position), piece=piece):
                                print(f"Found {piece} to block or capture at {self.position_to_string(position)}, targeting {self.position_to_string(target_position)}")
                                return True
                        except Exception:  # Catch any exception
                            continue
        return False
    
    def can_capture_or_block(self, checking_piece_position, side="w"):
        """ Determine if any piece of the specified side can capture the checking piece.
        Parameters:
            checking_piece_position (int): The position of the checking piece.
            side (str): The side ('w' for white, 'b' for black) of the pieces.
        Returns:
            bool: True if any piece can capture the checking piece, False otherwise.
        """
        # get king position
        king_position = self.get_king_position(side=side)
        rank_diff = (king_position // 8) - (checking_piece_position // 8)
        file_diff = (king_position % 8) - (checking_piece_position % 8)

        # Determine the check path
        if rank_diff == 0:
            step = 1 if file_diff > 0 else -1
            path_positions = [(king_position // 8) * 8 + file for file in range(checking_piece_position % 8 + step, king_position % 8, step)]
        elif file_diff == 0:
            step = 8 if rank_diff > 0 else -8
            path_positions = [rank * 8 + (king_position % 8) for rank in range(checking_piece_position // 8 + step // 8, king_position // 8, step // 8)]
        elif abs(rank_diff) == abs(file_diff):
            rank_step = 1 if rank_diff > 0 else -1
            file_step = 1 if file_diff > 0 else -1
            path_positions = []
            rank = checking_piece_position // 8 + rank_step
            file = checking_piece_position % 8 + file_step
            while 0 <= rank < 8 and 0 <= file < 8 and rank * 8 + file != king_position:
                path_positions.append(rank * 8 + file)
                rank += rank_step
                file += file_step
        else:
            # If the check is not along a rank, file, or diagonal, no path exists
            return False

        path_positions.append(checking_piece_position)  # Include the position of the checking piece
        
        # Check if any piece can block or capture along the path
        return any(self.can_piece_block_or_capture(position, side=side) for position in path_positions)
    
    def side_is_checkmated(self, side="w"):
        checking_piece_position = self.determine_if_checked(side=side)
        if checking_piece_position:
            print("Checking if king can move")
            if not self.king_can_move(side=side):
                print("King can't move")
                print("Checking if any piece can block or capture")
                if self.can_capture_or_block(checking_piece_position, side=side):
                    return False
                else:
                    return True
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
    
    def get_all_possible_moves_current_pos(self, side):
        """
        Get all possible moves for the current position of the specified side.
        :param self: The Chessboard object.
        :param side: The side for which to calculate the possible moves.
        :return: A dictionary containing the possible moves for each piece of the specified side.
        """
        possible_moves = {"wP": {}, "wN": {}, "wB": {}, "wR": {}, "wQ": {}, "wK": {}} if side == "w" else {
            "bP": {}, "bN": {}, "bB": {}, "bR": {}, "bQ": {}, "bK": {}}
        # Get bishop moves
        bishop_bitboard = self.board[side + 'B'] 
        for position in range(64):
            if bishop_bitboard & (1 << position):
                possible_moves_bitboard = self.B_lookup[position]
                possible_moves[side + 'B'][position] = possible_moves_bitboard
        
        # Get knight moves
        knight_bitboard = self.board[side + 'N']
        for position in range(64):
            if knight_bitboard & (1 << position):
                possible_moves_bitboard = self.N_lookup[position]
                possible_moves[side + 'N'][position] = possible_moves_bitboard
        
        # Get rook moves
        rook_bitboard = self.board[side + 'R']
        for position in range(64):
            if rook_bitboard & (1 << position):
                possible_moves_bitboard = self.R_lookup[position]
                possible_moves[side + 'R'][position] = possible_moves_bitboard
        
        # Get queen moves
        queen_bitboard = self.board[side + 'Q']
        for position in range(64):
            if queen_bitboard & (1 << position):
                possible_moves_bitboard = self.Q_lookup[position]
                possible_moves[side + 'Q'][position] = possible_moves_bitboard
        
        # Get king moves
        king_position = self.get_king_position(side)
        possible_moves_bitboard = 0
        for move in self.get_adjacent_positions(king_position):
            if not self.determine_if_king_cant_move_here(move, side):
                possible_moves_bitboard |= (1 << move)

        # Check for castling moves and add them if possible
        if self.apply_castling_constraints(side, "k"):  # Kingside castling
            if side == "w":
                possible_moves_bitboard |= (1 << 62)  # g1 for white
            else:
                possible_moves_bitboard |= (1 << 6)   # g8 for black
        if self.apply_castling_constraints(side, "q"):  # Queenside castling
            if side == "w":
                possible_moves_bitboard |= (1 << 58)  # c1 for white
            else:
                possible_moves_bitboard |= (1 << 2)   # c8 for black

        for position in range(64):
            if possible_moves_bitboard & (1 << position):
                possible_moves[side + 'K'][position] = possible_moves_bitboard
        
        # get pawn moves
        pawn_bitboard = self.board[side + 'P']
        for position in range(64):
            if pawn_bitboard & (1 << position):
                possible_moves_bitboard = self.P_lookup_w[position] if side == 'w' else self.P_lookup_b[position]
                possible_moves[side + 'P'][position] = possible_moves_bitboard
                
        return possible_moves
        
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
        follows_constraint = False
        if self.white_to_move and piece[0] == "w" or not self.white_to_move and piece[0] == "b" or "O" in piece:
        
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
    def piece_castle_conditional(self, piece):
        if piece is None:
            return True
        elif "O" in piece:
            return False
        elif "O" not in piece:
            return True
        
    def move_piece(self, from_square="", to_square="", piece=None):
        old_self = ConstrainedGameState()
        old_board = self.board.copy()
        for attr, value in vars(self).items():
            if isinstance(value, (bool, list, np.ndarray)):
                setattr(old_self, attr, deepcopy(value))
        if piece is not None and len(piece) == 1:
            if self.white_to_move:
                piece = 'w' + piece
            else:
                piece = 'b' + piece
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
                    raise ValueError(f"No {piece} at {from_square}.")
                promotion = False
                if 'P' in moving_piece and ((moving_piece.startswith('w') and to_position // 8 == 0) or (moving_piece.startswith('b') and to_position // 8 == 7)):
                    promotion = True
                    self.promoting = True
                else:
                    self.promoting = False
                if self.piece_constrainer(from_square, to_square, piece=moving_piece):
                    self.clear_piece(moving_piece, from_position)
                    if not promotion:
                        self.set_piece(moving_piece, to_position)
                    if promotion:
                        self.promoting_from_pos = from_position
                        self.promoting_to_pos = to_position
                    self.king_or_rook_moved(moving_piece, from_square=from_square)
                    self.P_lookup_w = self.get_all_possible_pawn_moves_for_side('w')
                    self.P_lookup_b = self.get_all_possible_pawn_moves_for_side('b')
                    
                    self.update_board_state()
                    self.white_to_move = not self.white_to_move
                    self.assign_checking_and_final_states()
                    
                    if self.white_to_move and self.white_in_check and not self.checkmated:
                        print("White is in check.")
                    if not self.white_to_move and self.black_in_check and not self.checkmated:
                        print("Black is in check.")
                    
                    self.check_resolved()
                    if not self.white_to_move and self.white_in_check:
                        raise ValueError("Illegal move. White is still in check or is put in check.")
                    if self.white_to_move and self.black_in_check:
                        raise ValueError("Illegal move. Black is still in check or is put in check.")
                    if not self.promoting:
                        move_array = np.array([self.piece_enum[moving_piece], from_position, to_position], dtype=np.int8)
                        self.move_log = np.vstack([self.move_log, move_array])
                    if self.first_move:
                        self.move_log = self.move_log[1:]
                        self.first_move = False
                else:
                    raise ValueError(f"Illegal move from {from_square} to {to_square} for {moving_piece}.")
            else:
                if self.piece_constrainer(piece=piece):
                    self.king_or_rook_moved(moving_piece=piece)
                    self.castling_logic(piece)
                    self.P_lookup_w = self.get_all_possible_pawn_moves_for_side('w')
                    self.P_lookup_b = self.get_all_possible_pawn_moves_for_side('b')
                    self.update_board_state()
                    
        except Exception as e:
            for attr in vars(old_self):
                setattr(self, attr, getattr(old_self, attr))
            self.board = old_board
            print(f"An error occurred while moving the piece: {e}")
            traceback.print_exc(file=sys.stdout)
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



