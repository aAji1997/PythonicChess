from ChessLogic import ConstrainedGameState, InvalidBishopMoveError, InvalidRookMoveError, InvalidQueenMoveError
import warnings
from copy import deepcopy
import hashlib
import time
import os
from BookIntegration import create_python_chess_integration

class GameEngine(ConstrainedGameState):
    def __init__(self):
        super().__init__()
        self.max_depth = 50
        self.history_table = {}  # For move ordering
        self.killer_moves = [[None] * 50 for _ in range(2)]  # Two killer moves per ply
        
        self.value_map = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 20000}
        
        # Initialize lookup tables for piece moves
        self.N_lookup = self.initialize_knight_moves()
        self.B_lookup = self.initialize_bishop_moves()
        self.R_lookup = self.initialize_rook_moves()
        self.Q_lookup = self.initialize_queen_moves()
        
        # Piece-square tables for positional evaluation
        self.pawn_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        
        self.knight_table = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]
        
        self.bishop_table = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ]
        
        self.rook_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        ]
        
        self.queen_table = [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ]
        
        self.king_middle_table = [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            20, 20,  0,  0,  0,  0, 20, 20,
            20, 30, 10,  0,  0, 10, 30, 20
        ]
        
        self.king_end_table = [
            -50,-40,-30,-20,-20,-30,-40,-50,
            -30,-20,-10,  0,  0,-10,-20,-30,
            -30,-10, 20, 30, 30, 20,-10,-30,
            -30,-10, 30, 40, 40, 30,-10,-30,
            -30,-10, 30, 40, 40, 30,-10,-30,
            -30,-10, 20, 30, 30, 20,-10,-30,
            -30,-30,  0,  0,  0,  0,-30,-30,
            -50,-30,-30,-30,-30,-30,-30,-50
        ]
        # Opening book integration
        book_path = os.path.join(os.path.dirname(__file__), 'book', 'Perfect2023.bin')
        self.opening_book = create_python_chess_integration(book_path)
    
    def initialize_knight_moves(self):
        """Initialize lookup table for all possible knight moves from each square."""
        knight_moves = {}
        for i in range(64):
            x, y = divmod(i, 8)
            
            # Calculate all possible knight moves for this position
            possible_moves_bitboard = 0
            knight_offsets = [
                (-2, -1), (-2, 1),  # Up 2, left/right 1
                (2, -1), (2, 1),    # Down 2, left/right 1
                (-1, -2), (-1, 2),  # Up 1, left/right 2
                (1, -2), (1, 2)     # Down 1, left/right 2
            ]
            
            # Calculate all theoretically possible moves for this position
            for dx, dy in knight_offsets:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < 8 and 0 <= new_y < 8:
                    target_pos = new_x * 8 + new_y
                    possible_moves_bitboard |= 1 << target_pos
            
            # Store all theoretical moves in lookup
            knight_moves[i] = possible_moves_bitboard
        
        return knight_moves
    
    def initialize_bishop_moves(self):
        """Initialize lookup table for all possible bishop moves from each square."""
        bishop_moves = {}
        
        # Calculate moves for all squares
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
        
        return bishop_moves
    
    def initialize_rook_moves(self):
        """Initialize lookup table for all possible rook moves from each square."""
        rook_moves = {}
        
        # Calculate moves for all squares
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
        
        return rook_moves
    
    def initialize_queen_moves(self):
        """Initialize lookup table for all possible queen moves from each square."""
        queen_moves = {}
        
        # Calculate moves for all squares
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
        
        return queen_moves
        
    def get_filtered_knight_moves(self, position, side, board):
        """Get filtered knight moves for a specific position and board state."""
        # Get theoretical moves from lookup
        possible_moves = self.N_lookup[position]
        
        # Get own pieces to avoid moving to squares occupied by own pieces
        own_pieces = 0
        for piece in ['P', 'R', 'N', 'B', 'Q', 'K']:
            own_pieces |= board.get(side + piece, 0)
        
        # Filter out moves to squares occupied by own pieces
        return possible_moves & ~own_pieces
    
    def get_filtered_bishop_moves(self, position, side, board):
        """Get filtered bishop moves for a specific position and board state."""
        # Get theoretical moves from lookup
        possible_moves = self.B_lookup[position]
        
        # Get own and opponent pieces
        own_pieces = 0
        opp_pieces = 0
        for piece in ['P', 'R', 'N', 'B', 'Q', 'K']:
            own_pieces |= board.get(side + piece, 0)
            opp_pieces |= board.get(('b' if side == 'w' else 'w') + piece, 0)
        
        filtered_moves = 0
        x, y = divmod(position, 8)
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
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
        
        return filtered_moves
    
    def get_filtered_rook_moves(self, position, side, board):
        """Get filtered rook moves for a specific position and board state."""
        # Get theoretical moves from lookup
        possible_moves = self.R_lookup[position]
        
        # Get own and opponent pieces
        own_pieces = 0
        opp_pieces = 0
        for piece in ['P', 'R', 'N', 'B', 'Q', 'K']:
            own_pieces |= board.get(side + piece, 0)
            opp_pieces |= board.get(('b' if side == 'w' else 'w') + piece, 0)
        
        filtered_moves = 0
        x, y = divmod(position, 8)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, left, down, up
        
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
        
        return filtered_moves
    
    def get_filtered_queen_moves(self, position, side, board):
        """Get filtered queen moves for a specific position and board state."""
        # Get theoretical moves from lookup
        possible_moves = self.Q_lookup[position]
        
        # Get own and opponent pieces
        own_pieces = 0
        opp_pieces = 0
        for piece in ['P', 'R', 'N', 'B', 'Q', 'K']:
            own_pieces |= board.get(side + piece, 0)
            opp_pieces |= board.get(('b' if side == 'w' else 'w') + piece, 0)
        
        filtered_moves = 0
        x, y = divmod(position, 8)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0),  # Rook directions
                     (-1, -1), (-1, 1), (1, -1), (1, 1)] # Bishop directions
        
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
        
        return filtered_moves
    
    def set_piece(self, piece, position, board):
        """Set a piece on the board, handling captures properly."""
        # Get the side of the piece being placed
        side = piece[0]  # 'w' or 'b'
        
        # Check if the target square is occupied by any piece
        for existing_piece in list(board.keys()):
            if board[existing_piece] & (1 << position):
                # If occupied by opponent piece, remove it (capture)
                if existing_piece[0] != side:
                    board[existing_piece] &= ~(1 << position)
                # If occupied by own piece, that's an error
                else:
                    raise ValueError(f"Position {position} is already occupied by a piece of the same color.")
        
        # Set the piece in its new position
        board[piece] |= (1 << position)
        return board

    def clear_piece(self, piece, position, board):
        board[piece] &= ~(1 << position)
    def get_piece_at_square(self, board, square):
        for piece, bitboard in board.items():
            if bitboard & (1 << self.string_to_position(square)):
                return piece
        return None
    
    def get_piece_at_position(self, board, position):
        for piece, bitboard in board.items():
            if bitboard & (1 << position):
                return piece
        return None
    
    def get_bishop_obstacles(self, board, from_position, to_position):
        ##} to {self.position_to_string(to_position)}")
        # Calculate the direction of the diagonal
        rank_diff = (to_position // 8) - (from_position // 8)
        ##
        file_diff = (to_position % 8) - (from_position % 8)
        ##
        
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
        for piece, bitboard in board.items():
            if bitboard & path_without_obstacles:
                return True  # Obstacle found, return True immediately
        
        return False  # No obstacles found
    def apply_bishop_constraints(self, board, from_position, to_position):
        """
        Apply bishop constraints to the move from one position to another.

        Args:
            from_position: The position from which the move starts.
            to_position: The position to which the move ends.

        Returns:
            bool: True if the move satisfies the bishop constraints, otherwise False.
        """
        # Check if the path is clear
        obstacles = self.get_bishop_obstacles(board, from_position, to_position)

        # Check if there are obstacles in the path
        if obstacles:
            from_square = self.position_to_string(from_position)
            to_square = self.position_to_string(to_position)
            raise ValueError(f"Invalid move for bishop: path from {from_square} to {to_square} is obstructed")

        return True

    def get_king_position(self, board, side="w"):
        """Get the position of the king for the specified side."""
        try:
            king_piece = side + 'K'
            king_bitboard = board.get(king_piece, 0)
            if king_bitboard == 0:
                return None
                
            position = 0
            while king_bitboard:
                if king_bitboard & 1:
                    return position
                king_bitboard >>= 1
                position += 1
            
            return None
        except Exception as e:
            return None

    def checking_queen_or_rook(self, king_position, side="w", board=None):
        """
        Determine if there is a rook or queen checking the king.
        
        Args:
            king_position (int): The position of the king on the board.
            side (str): The side to check for rook or queen (default is "w" for white).
            board (dict): Optional board state to check. If None, uses self.board.
        Returns:
            int or None: The position of the checking piece if found, None otherwise.
        """
        if board is None:
            board = self.board
            
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
                    piece = self.get_piece_at_position(board, position)
                    if piece:
                        if piece in rook_queen_pieces and not self.get_rook_obstacles(from_position=position, to_position=king_position, board=board):
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
                    piece = self.get_piece_at_position(board, position)
                    if piece:
                        if piece in rook_queen_pieces and not self.get_rook_obstacles(from_position=position, to_position=king_position, board=board):
                            return position  # Return the position of the checking piece
                        break  # Stop if any piece is encountered
                else:
                    break  # Stop if edge of board is reached

        return None  # Return None if no checking piece is found

    def checking_bishop_or_queen(self, king_position, side="w", board=None):
        """
        Determine if the opponent has a bishop or queen that can attack the king at the given position.
        
        Args:
            king_position (int): The position of the king on the board.
            side (str): The side to check for bishop or queen (default is "w" for white).
            board (dict): Optional board state to check. If None, uses self.board.
        Returns:
            int or None: The position of the checking piece if found, None otherwise.
        """
        if board is None:
            board = self.board
            
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
                    piece = self.get_piece_at_position(board, new_position)

                    # If a piece is encountered, check if it's a bishop or queen
                    if piece in bishop_queen_pieces:
                        return new_position
                    elif piece:  # Any other piece blocks the check
                        break
                else:  # Reached the edge of the board
                    break

                current_position = new_position

        return None  # No bishop or queen checking the king

    def checking_knight(self, king_position, side="w", board=None):
        """
        Check if an opponent's knight is checking the king.
        
        Args:
            king_position (int): The position of the king on the board.
            side (str): The side to check for knight (default is "w" for white).
            board (dict): Optional board state to check. If None, uses self.board.
        Returns:
            int or None: The position of the checking knight if found, None otherwise.
        """
        if board is None:
            board = self.board
            
        # Determine the opposite side
        opposite_side = 'b' if side == 'w' else 'w'
        knight_moves = self.N_lookup[king_position]
        opposite_knight = opposite_side + 'N'
        
        for move in range(64):
            if knight_moves & (1 << move) and self.get_piece_at_position(board, move) == opposite_knight:
                return move
        return None

    def checking_pawn(self, king_position, side="w", board=None):
        """
        Check if an opponent's pawn is checking the king.
        
        Args:
            king_position (int): The position of the king on the board.
            side (str): The side to check for pawns (default is "w" for white).
            board (dict): Optional board state to check. If None, uses self.board.
        Returns:
            int or None: The position of the checking pawn if found, None otherwise.
        """
        if board is None:
            board = self.board
            
        # Determine the opposite side
        opposite_side = 'b' if side == 'w' else 'w'
        
        # Check the diagonals for a pawn
        # White pawns attack downward, black pawns attack upward
        pawn_moves = [9, 7] if opposite_side == 'w' else [-9, -7]
        
        for move in pawn_moves:
            check_position = king_position + move
            check_rank, check_file = divmod(check_position, 8)
            king_rank, king_file = divmod(king_position, 8)
            
            # Ensure the move doesn't wrap around the board
            if 0 <= check_rank < 8 and 0 <= check_file < 8:
                # Check that we didn't cross more than one file
                if abs(check_file - king_file) == 1:
                    if self.get_piece_at_position(board, check_position) == opposite_side + 'P':
                        return check_position
                    
        return None
                    
    def determine_if_checked(self, board, side="w"):
        """Determine if the specified side's king is checked."""
        king_position = self.get_king_position(board, side)
        
        # Check each type of threat
        adj_k = self.are_kings_adjacent(side, king_position, board)
        if adj_k:
            return adj_k
            
        QR_check = self.checking_queen_or_rook(king_position, side, board)
        if QR_check:
            return QR_check
            
        BQ_check = self.checking_bishop_or_queen(king_position, side, board)
        if BQ_check:
            return BQ_check
            
        N_check = self.checking_knight(king_position, side, board)
        if N_check:
            return N_check
            
        P_check = self.checking_pawn(king_position, side, board)
        if P_check:
            return P_check
            
        return False
    
    def are_kings_adjacent(self, side, king_position, board=None):
        """
        Check if the king is moving to a square adjacent to the opponent's king.
    
        Args:
            side (str): The side moving, either 'w' or 'b'.
            king_position (int): The position to check
            board (dict): Optional board state to check. If None, uses self.board.
    
        Returns: 
            bool: True if the kings are adjacent after moving, False otherwise.
        """
        if board is None:
            board = self.board
            
        # Get opponent's king position
        opponent_king_pos = self.get_king_position(board, 'b' if side == 'w' else 'w')
        
        # Get adjacent squares to opponent's king
        adj_squares = self.get_adjacent_positions(opponent_king_pos)
        
        # Check if target position is adjacent 
        return king_position in adj_squares

    def determine_if_king_cant_move_here(self, board, new_king_position, side="w"):
        """
        Determine if the king cannot move to the given position by checking for threats.
        
        Args:
            board (dict): The board state to check
            new_king_position (int): The new position of the king on the board
            side (str): The side of the player, default is "w" for white
            
        Returns:
            bool: False if the king is not threatened, otherwise returns the threatening piece position
        """
        # Save current board state
        curr_king_position = self.get_king_position(board, side)
        if curr_king_position is None:
            #
            return True  # If we can't find the king, consider the move invalid
            
        # Create a copy of the board for testing
        test_board = board.copy()
        
        # Temporarily move the king
        piece = side + "K"
        self.clear_piece(piece, curr_king_position, test_board)
        try:
            self.set_piece(piece, new_king_position, test_board)
        except ValueError:
            return True  # If we can't set the piece, the move is invalid
        
        # Perform checks
        adj_k = self.are_kings_adjacent(side, new_king_position, test_board)
        if adj_k:
            return adj_k

        QR_check = self.checking_queen_or_rook(new_king_position, side, test_board)
        if QR_check:
            return QR_check

        BQ_check = self.checking_bishop_or_queen(new_king_position, side, test_board)
        if BQ_check:
            return BQ_check

        N_check = self.checking_knight(new_king_position, side, test_board)
        if N_check:
            return N_check

        P_check = self.checking_pawn(new_king_position, side, test_board)
        if P_check:
            return P_check

        return False
    def king_can_move(self, board, side="w"):
        """
        Checks if the king can move to any adjacent positions.
        
        Args:
            board (dict): The board state to check
            side (str): The side of the king, default is "w" (white)
            
        Returns:
            bool: True if the king can move, False otherwise
        """
        king_position = self.get_king_position(board, side)
        adj_positions = self.get_adjacent_positions(king_position)
        non_moveable_positions = []
        for position in adj_positions:
            occupying_piece = self.get_piece_at_position(board, position)
            if occupying_piece is not None and occupying_piece[0] == side:
                non_moveable_positions.append(position) # piece is present at position, therefore king can't move here
            elif self.determine_if_king_cant_move_here(board, position, side):
                non_moveable_positions.append(position) # king is in check here, therefore king can't move here

        # If all adjacent positions are non-moveable, the king cannot move
        if set(adj_positions) == set(non_moveable_positions):
            return False
        else:
            return True
    
    def can_piece_block_or_capture(self, target_position, side="w", board=None):
        """
        Check if any piece can block or capture at the target position.
        
        Args:
            target_position (int): The position to check for blocking/capturing
            side (str): The side to check ('w' for white, 'b' for black)
            board (dict): Optional board state to check. If None, uses self.board.
            
        Returns:
            bool: True if a piece can block/capture, False otherwise
        """
        if board is None:
            board = self.board
            
        # Get the position of the side's own king to skip it
        own_king_position = self.get_king_position(board, side)
        
        # Check for any piece that can reach the target position
        for piece, bitboard in board.items():
            if piece.startswith(side) and not piece.endswith('K'):  # Skip the king
                for position in range(64):
                    if bitboard & (1 << position) and position != own_king_position:  # Skip the king's position
                        try:
                            # For pawns, validate diagonal moves strictly
                            if piece.endswith('P'):
                                from_file = position % 8
                                to_file = target_position % 8
                                from_rank = position // 8
                                to_rank = target_position // 8
                                
                                # If files are different (diagonal move)
                                if from_file != to_file:
                                    # Must have a piece to capture
                                    target_piece = self.get_piece_at_position(board, target_position)
                                    if not target_piece:
                                        continue
                                    # Must be moving in correct direction
                                    if (side == 'w' and to_rank >= from_rank) or (side == 'b' and to_rank <= from_rank):
                                        continue
                                # If files are same (straight move)
                                else:
                                    # Cannot capture straight ahead
                                    target_piece = self.get_piece_at_position(board, target_position)
                                    if target_piece:
                                        continue
                                    # Must be moving in correct direction
                                    if (side == 'w' and to_rank >= from_rank) or (side == 'b' and to_rank <= from_rank):
                                        continue
                            
                            # Save current board state
                            old_board = deepcopy(board)
                            
                            try:
                                # Try the move
                                if self.piece_constrainer(from_position=position, to_position=target_position, piece=piece, board=board):
                                    # Actually make the move
                                    self.clear_piece(piece, position, board)
                                    self.set_piece(piece, target_position, board)
                                    
                                    # Check if this resolves the check
                                    if not self.determine_if_checked(side=side, board=board):
                                        # Restore board state
                                        board = old_board
                                        return True
                            except Exception:  # Catch any exception
                                pass
                            finally:
                                # Always restore board state
                                board = old_board
                                
                        except Exception:  # Catch any exception in the outer try block
                            continue
        return False
    

    def can_capture_or_block(self, checking_piece_position, side="w", board=None):
        """ Determine if any piece of the specified side can capture the checking piece.
        Parameters:
            checking_piece_position (int): The position of the checking piece.
            side (str): The side ('w' for white, 'b' for black) of the pieces.
            board (dict): Optional board state to check. If None, uses self.board.
        Returns:
            bool: True if any piece can capture the checking piece, False otherwise.
        """
        if board is None:
            board = self.board
            
        # get king position
        king_position = self.get_king_position(side=side, board=board)
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
        return any(self.can_piece_block_or_capture(position, side=side, board=board) for position in path_positions)
    
    def get_all_possible_moves_current_pos(self, side, board=None):
        """Get all possible moves for the current position of the specified side."""
        if board is None:
            board = self.board

        possible_moves = {side + "P": {}, side + "N": {}, side + "B": {}, side + "R": {}, side + "Q": {}, side + "K": {}}

        # Get own and opponent pieces
        own_pieces = 0
        opponent_pieces = 0
        all_pieces = 0
        for piece in ['P', 'R', 'N', 'B', 'Q', 'K']:
            own_pieces |= board.get(side + piece, 0)
            opponent_pieces |= board.get(('b' if side == 'w' else 'w') + piece, 0)
        all_pieces = own_pieces | opponent_pieces

        # Get pawn moves
        pawn_bitboard = board.get(side + 'P', 0)
        position = 0
        while pawn_bitboard:
            if pawn_bitboard & 1:
                possible_moves_bitboard = 0
                x, y = divmod(position, 8)
                
                # Determine direction and starting rank based on side
                direction = -1 if side == 'w' else 1
                starting_rank = 6 if side == 'w' else 1
                
                # Single square forward move
                new_rank = x + direction
                if 0 <= new_rank < 8:
                    target_pos = new_rank * 8 + y
                    target_bit = 1 << target_pos
                    
                    # If square is empty
                    if not (target_bit & all_pieces):
                        possible_moves_bitboard |= target_bit
                        
                        # Two square forward move from starting position
                        if x == starting_rank:
                            new_rank = x + 2 * direction
                            if 0 <= new_rank < 8:
                                target_pos = new_rank * 8 + y
                                target_bit = 1 << target_pos
                                # Only if both squares are empty
                                if not (target_bit & all_pieces):
                                    possible_moves_bitboard |= target_bit

                # Capture moves (diagonal)
                for capture_file in [y - 1, y + 1]:
                    if 0 <= capture_file < 8:  # Check file bounds
                        new_rank = x + direction
                        if 0 <= new_rank < 8:  # Check rank bounds
                            target_pos = new_rank * 8 + capture_file
                            target_bit = 1 << target_pos
                            
                            # Regular capture
                            if target_bit & opponent_pieces:
                                possible_moves_bitboard |= target_bit
                            
                            # En passant capture
                            elif target_pos == self.en_passant_target:
                                possible_moves_bitboard |= target_bit
                
                if possible_moves_bitboard:
                    possible_moves[side + 'P'][position] = possible_moves_bitboard
            
            pawn_bitboard >>= 1
            position += 1

        # Get knight moves
        knight_bitboard = board.get(side + 'N', 0)
        position = 0
        while knight_bitboard:
            if knight_bitboard & 1:
                filtered_moves = self.get_filtered_knight_moves(position, side, board)
                if filtered_moves:
                    possible_moves[side + 'N'][position] = filtered_moves
            knight_bitboard >>= 1
            position += 1

        # Get bishop moves
        bishop_bitboard = board.get(side + 'B', 0)
        position = 0
        while bishop_bitboard:
            if bishop_bitboard & 1:
                filtered_moves = self.get_filtered_bishop_moves(position, side, board)
                if filtered_moves:
                    possible_moves[side + 'B'][position] = filtered_moves
            bishop_bitboard >>= 1
            position += 1

        # Get rook moves
        rook_bitboard = board.get(side + 'R', 0)
        position = 0
        while rook_bitboard:
            if rook_bitboard & 1:
                filtered_moves = self.get_filtered_rook_moves(position, side, board)
                if filtered_moves:
                    possible_moves[side + 'R'][position] = filtered_moves
            rook_bitboard >>= 1
            position += 1

        # Get queen moves
        queen_bitboard = board.get(side + 'Q', 0)
        position = 0
        while queen_bitboard:
            if queen_bitboard & 1:
                filtered_moves = self.get_filtered_queen_moves(position, side, board)
                if filtered_moves:
                    possible_moves[side + 'Q'][position] = filtered_moves
            queen_bitboard >>= 1
            position += 1
        
        return possible_moves

    def convert_moves_to_legal(self, possible_moves, board=None):
        """Convert possible moves to legal moves by checking constraints."""
        if board is None:
            board = self.board
        
        ##
        ##
        
        legal_moves = []
        side = "w" if self.white_to_move else "b"
        
        for piece_type in possible_moves:
            #
            for from_position, moves_bitboard in possible_moves[piece_type].items():
                #}")
                
                # Convert bitboard to list of positions
                for to_position in range(64):
                    if moves_bitboard & (1 << to_position):
                        from_square = self.position_to_string(from_position)
                        to_square = self.position_to_string(to_position)
                        #
                        
                        try:
                            # Check if move is legal
                            if self.piece_constrainer(from_position=from_position, to_position=to_position, piece=piece_type, board=board):
                                legal_moves.append((from_position, to_position))
                        except Exception as e:
                            #}")
                            continue
        
        #
        return legal_moves

    def move_piece(self, piece, from_position, to_position, board):
            self.clear_piece(piece, from_position, board)
            self.set_piece(piece, to_position, board)
            
            return board

        
    def evaluate_board(self, board):
        """Evaluate the current board position."""
        if board is None:
            board = self.board
            
        # Calculate material balance
        material_score = 0
        white_pawns = bin(board.get('wP', 0)).count('1')
        black_pawns = bin(board.get('bP', 0)).count('1')
        
        for piece in ['P', 'N', 'B', 'R', 'Q']:
            white_count = bin(board.get('w' + piece, 0)).count('1')
            black_count = bin(board.get('b' + piece, 0)).count('1')
            material_score += (white_count - black_count) * self.value_map[piece]
            
        #")
        
        # Calculate position score
        position_score = 0
        for piece in ['P', 'N', 'B', 'R', 'Q', 'K']:
            # White pieces
            white_pieces = board.get('w' + piece, 0)
            for pos in range(64):
                if white_pieces & (1 << pos):
                    if piece == 'P':
                        position_score += self.pawn_table[pos]
                    elif piece == 'N':
                        position_score += self.knight_table[pos]
                    elif piece == 'B':
                        position_score += self.bishop_table[pos]
                        
            # Black pieces (mirror the tables)
            black_pieces = board.get('b' + piece, 0)
            for pos in range(64):
                if black_pieces & (1 << pos):
                    mirror_pos = pos ^ 56  # Mirror position for black
                    if piece == 'P':
                        position_score -= self.pawn_table[mirror_pos]
                    elif piece == 'N':
                        position_score -= self.knight_table[mirror_pos]
                    elif piece == 'B':
                        position_score -= self.bishop_table[mirror_pos]
                        
        #
        
        # Calculate pawn structure score
        pawn_structure_score = self.evaluate_pawn_structure(board) * 2
        #
        
        # Calculate mobility score
        mobility_score = self.evaluate_mobility(board)
        #
        
        # King safety evaluation
        white_king_safety = self.evaluate_king_safety(board, 'w')
        black_king_safety = self.evaluate_king_safety(board, 'b')
        king_safety_score = white_king_safety - black_king_safety
        #
        
        # Final score calculation
        # Material is most important, followed by king safety
        score = material_score * 2 + king_safety_score + position_score + pawn_structure_score + mobility_score
        
        #
        return score
        
    def evaluate_king_safety(self, board, side):
        """Evaluate king safety for the given side."""
        king_pos = self.get_king_position(board, side)
        if king_pos is None:
            return 0
            
        safety_score = 0
        
        # Check if king is in check
        if self.determine_if_checked(board, side):
            safety_score -= 100
            
        # Count defended squares around king
        adj_squares = self.get_adjacent_positions(king_pos)
        for square in adj_squares:
            if self.is_square_protected(board, square, side):
                safety_score += 10
                
        # Bonus for castled position
        rank = king_pos // 8
        file = king_pos % 8
        if (side == 'w' and rank == 7) or (side == 'b' and rank == 0):
            if file in [2, 6]:  # Castled position
                safety_score += 50
                
        return safety_score
        
    def evaluate_pawn_structure(self, board):
        """Evaluate pawn structure including doubled, isolated, and passed pawns."""
        score = 0
        
        # Get pawn positions
        white_pawns = []
        black_pawns = []
        wp_bitboard = board.get('wP', 0)
        bp_bitboard = board.get('bP', 0)
        
        position = 0
        while wp_bitboard:
            if wp_bitboard & 1:
                white_pawns.append(position)
            wp_bitboard >>= 1
            position += 1
            
        position = 0
        while bp_bitboard:
            if bp_bitboard & 1:
                black_pawns.append(position)
            bp_bitboard >>= 1
            position += 1
        
        # for p in white_pawns]}")
        # for p in black_pawns]}")
        
        # Evaluate doubled pawns (severe penalty)
        doubled_pawn_score = 0
        for file in range(8):
            wp_in_file = sum(1 for p in white_pawns if p % 8 == file)
            bp_in_file = sum(1 for p in black_pawns if p % 8 == file)
            
            if wp_in_file > 1:
                doubled_pawn_score -= 30 * (wp_in_file - 1)
                #}: -{30 * (wp_in_file - 1)}")
            if bp_in_file > 1:
                doubled_pawn_score += 30 * (bp_in_file - 1)
                #}: +{30 * (bp_in_file - 1)}")
        
        score += doubled_pawn_score
        
        # Evaluate isolated pawns (significant penalty)
        for file in range(8):
            # Check adjacent files for pawns
            wp_in_file = any(p % 8 == file for p in white_pawns)
            wp_adjacent = any(p % 8 in [file-1, file+1] for p in white_pawns if 0 <= p % 8 < 8)
            
            bp_in_file = any(p % 8 == file for p in black_pawns)
            bp_adjacent = any(p % 8 in [file-1, file+1] for p in black_pawns if 0 <= p % 8 < 8)
            
            if wp_in_file and not wp_adjacent:
                score -= 25  # Increased penalty for isolated pawns
                #}: -25")
            if bp_in_file and not bp_adjacent:
                score += 25
                #}: +25")
        
        # Evaluate protected pawns (bonus for protection)
        for pawn_pos in white_pawns:
            if self.is_square_protected(board, pawn_pos, 'w'):
                score += 15  # Bonus for protected pawns
                #}: +15")
        
        for pawn_pos in black_pawns:
            if self.is_square_protected(board, pawn_pos, 'b'):
                score -= 15  # Also bonus for black protected pawns (negative because black's perspective)
                #}: -15")
        
        # Evaluate passed pawns (significant bonus)
        for pawn_pos in white_pawns:
            file = pawn_pos % 8
            rank = pawn_pos // 8
            is_passed = True
            # Check if any black pawns can block or capture
            for bp_pos in black_pawns:
                bp_file = bp_pos % 8
                bp_rank = bp_pos // 8
                if bp_file in [file-1, file, file+1] and bp_rank < rank:
                    is_passed = False
                    break
            if is_passed:
                bonus = (7 - rank) * 10  # Bigger bonus for more advanced pawns
                score += bonus
                #}: +{bonus}")
        
        for pawn_pos in black_pawns:
            file = pawn_pos % 8
            rank = pawn_pos // 8
            is_passed = True
            # Check if any white pawns can block or capture
            for wp_pos in white_pawns:
                wp_file = wp_pos % 8
                wp_rank = wp_pos // 8
                if wp_file in [file-1, file, file+1] and wp_rank > rank:
                    is_passed = False
                    break
            if is_passed:
                bonus = rank * 10  # Bigger bonus for more advanced pawns
                score -= bonus
                #}: -{bonus}")
        
        #
        return score
        
    def evaluate_mobility(self, board):
        """Evaluate piece mobility (number of legal moves available)."""
        white_mobility = len(self.get_all_legal_moves('w', board))
        black_mobility = len(self.get_all_legal_moves('b', board))
        
        return white_mobility - black_mobility

    def piece_constrainer(self, from_position=0, to_position=0, piece="wP", board=None):
        """Check if a move follows the piece's movement constraints."""
        if board is None:
            board = self.board

        follows_constraint = False
        
        # Check if it's the right color's turn to move
        if (piece[0] == "w" and self.white_to_move) or (piece[0] == "b" and not self.white_to_move):
            # Check if the destination square is occupied by a piece of the same color
            target_piece = self.get_piece_at_position(board, to_position)
            if target_piece and target_piece[0] == piece[0]:
                return False

            if piece[1] == "P":
                follows_constraint = self.apply_pawn_constraints(from_position=from_position, 
                                                              to_position=to_position, 
                                                              pawn_type=piece[0],
                                                              board=board)
            
            elif piece[1] == "B":
                follows_constraint = self.apply_bishop_constraints(from_position=from_position, 
                                                              to_position=to_position,
                                                              board=board)

            elif piece[1] == "N":
                follows_constraint = self.apply_knight_constraints(from_position=from_position, 
                                                              to_position=to_position,
                                                              board=board)
            
            elif piece[1] == "R":
                follows_constraint = self.apply_rook_constraints(from_position=from_position, 
                                                              to_position=to_position,
                                                              board=board)
            elif piece[1] == "Q":
                follows_constraint = self.apply_queen_constraints(from_position=from_position, 
                                                              to_position=to_position,
                                                              board=board)
            
            elif piece[1] == "K":
                follows_constraint = self.apply_king_constraints(from_position=from_position, 
                                                              to_position=to_position,
                                                              board=board)
            elif piece == "OO": # kingside castling
                if piece[0] == "w":
                    follows_constraint = self.apply_castling_constraints(side="w", direction="k", board=board)
                else:
                    follows_constraint = self.apply_castling_constraints(side="b", direction="k", board=board)
            elif piece == "OOO": # queenside castling
                if piece[0] == "w":
                    follows_constraint = self.apply_castling_constraints(side="w", direction="q", board=board)
                else:
                    follows_constraint = self.apply_castling_constraints(side="b", direction="q", board=board)
                    
            else:
                # log warning saying constraint isn't implemented
                follows_constraint = True
                warnings.warn(f"Constraint not implemented for {piece[1]}")
        else:
            raise ValueError(f"Not the right color to move: {piece}")
        
        return follows_constraint
    
    def order_moves(self, moves, board, ply):
        """Order moves for better alpha-beta pruning efficiency."""
        move_scores = []
        for move in moves:
            score = 0
            piece, from_pos, to_pos, promotion = move
            
            # Prioritize captures based on MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
            target_piece = self.get_piece_at_position(board, to_pos)
            if target_piece:
                # Higher weight for captures
                capture_value = self.value_map[target_piece[1]] * 10
                
                # Extra bonus for king captures of high-value pieces
                if piece[1] == 'K':
                    # Check if the king would be safe after the capture
                    test_board = self.make_move(board.copy(), move)
                    if test_board and not self.determine_if_checked(test_board, piece[0]):
                        capture_value *= 2  # Double the value for safe king captures
                
                score += capture_value
                # Bonus for safe captures (capturing piece is protected)
                if self.is_square_protected(board, to_pos, piece[0]):
                    score += 50
                # Additional bonus for capturing with lower value piece
                score += (self.value_map[target_piece[1]] - self.value_map[piece[1]]) * 2
                
            # Prioritize promotions
            if promotion:
                score += self.value_map[promotion] * 5
                
            # Bonus for protected pieces
            if self.is_square_protected(board, from_pos, piece[0]):
                score += 30
                
            # Use history heuristic
            move_key = (piece, from_pos, to_pos)
            score += self.history_table.get(move_key, 0)
            
            # Killer move bonus
            if move in self.killer_moves[ply % 2]:
                score += 40
                
            move_scores.append((score, move))
            
        # Sort moves by score in descending order
        move_scores.sort(reverse=True)
        return [move for score, move in move_scores]

    def quiescence_search(self, board, alpha, beta, depth, maximizing_player):
        """Quiescence search to handle tactical sequences."""
        stand_pat = self.evaluate_board(board)
        
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
            
        if depth == -4:  # Limit quiescence search depth
            return stand_pat
            
        # Get only captures and checks
        moves = self.get_all_legal_moves('w' if maximizing_player else 'b', board)
        captures = [move for move in moves if self.get_piece_at_position(board, move[2])]
        
        for move in captures:
            new_board = self.make_move(board, move)
            if new_board:
                score = -self.quiescence_search(new_board, -beta, -alpha, depth - 1, not maximizing_player)
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
                    
        return alpha

    def minmax(self, board, depth, alpha, beta, maximizing_player, ply=0):
        """Enhanced minimax with alpha-beta pruning, move ordering and quiescence."""
        if not isinstance(board, dict):
            return float('-inf') if maximizing_player else float('inf'), None
            
        if depth == 0:
            return self.quiescence_search(board, alpha, beta, 0, maximizing_player), None
            
        if self.is_game_over(board):
            return self.evaluate_board(board), None
    
        side = 'w' if maximizing_player else 'b'
        try:
            moves = self.get_all_legal_moves(side, board)
            # Debug only pawn moves and their evaluation
            pawn_moves = [move for move in moves if move[0] == f"{side}P"]
            if pawn_moves:
                ##} pawn moves at depth {depth}")
                pass
            
            moves = self.order_moves(moves, board, ply)  # Order moves for better pruning
        except Exception as e:
            #}")
            return float('-inf') if maximizing_player else float('inf'), None
            
        best_move = None
        if maximizing_player:
            max_eval = float('-inf')
            for move in moves:
                try:
                    new_board = self.make_move(board, move)
                    if move[0].endswith('P'):  # Debug pawn move evaluation
                        current_eval = self.evaluate_board(board)
                        ##} -> {self.position_to_string(move[2])}")
                        ##
                    eval, _ = self.minmax(new_board, depth - 1, alpha, beta, False, ply + 1)
                    
                    if eval > max_eval:
                        max_eval = eval
                        best_move = move
                        if move[0].endswith('P'):  # Debug best pawn move
                            ##} -> {self.position_to_string(move[2])} with eval {eval}")
                            pass
                        
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        if not self.get_piece_at_position(board, move[2]):  # If not a capture
                            self.killer_moves[ply % 2][1] = self.killer_moves[ply % 2][0]
                            self.killer_moves[ply % 2][0] = move
                        break
                        
                except Exception as e:
                    #}")
                    continue
                    
            # Update history table for the best move
            if best_move:
                move_key = (best_move[0], best_move[1], best_move[2])
                self.history_table[move_key] = self.history_table.get(move_key, 0) + depth * depth
                
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in moves:
                try:
                    new_board = self.make_move(board, move)
                    if move[0].endswith('P'):  # Debug pawn move evaluation
                        current_eval = self.evaluate_board(board)
                        ##} -> {self.position_to_string(move[2])}")
                        ##
                    eval, _ = self.minmax(new_board, depth - 1, alpha, beta, True, ply + 1)
                    
                    if eval < min_eval:
                        min_eval = eval
                        best_move = move
                        if move[0].endswith('P'):  # Debug best pawn move
                            ##} -> {self.position_to_string(move[2])} with eval {eval}")
                            pass
                        
                    beta = min(beta, eval)
                    if beta <= alpha:
                        if not self.get_piece_at_position(board, move[2]):
                            self.killer_moves[ply % 2][1] = self.killer_moves[ply % 2][0]
                            self.killer_moves[ply % 2][0] = move
                        break
                        
                except Exception as e:
                    #}")
                    continue
                    
            # Update history table for the best move
            if best_move:
                move_key = (best_move[0], best_move[1], best_move[2])
                self.history_table[move_key] = self.history_table.get(move_key, 0) + depth * depth
                
            return min_eval, best_move

    def iterative_deepening_search(self, board, max_time=5.0):
        """Iterative deepening search with time control."""
        # Attempt opening book move before search
        if hasattr(self, 'opening_book') and self.opening_book:
            book_move = self.opening_book.get_book_move(self)
            if book_move:
                # Convert book move to engine format
                from_square = book_move['from_square']
                to_square = book_move['to_square']
                from_pos = self.string_to_position(from_square)
                to_pos = self.string_to_position(to_square)
                piece = self.get_piece_at_position(self.board, from_pos)
                return (piece, from_pos, to_pos, None)
        start_time = time.time()
        best_move = None
        
        # Determine if engine is maximizing based on whose turn it is
        maximizing_player = self.white_to_move
        
        for depth in range(1, 7):  # Start with depth 1, increase gradually
            if time.time() - start_time > max_time:
                break
                
            try:
                _, move = self.minmax(board, depth, float('-inf'), float('inf'), maximizing_player)
                if move:  # Only update if we found a valid move
                    best_move = move
            except Exception:
                break
                
        return best_move

    def is_game_over(self, board):
        """Check if the game is over (checkmate or stalemate)."""
        if not isinstance(board, dict):
            return False
            
        try:
            # Check each end condition separately
            white_checkmate = self.is_checkmate(board, 'w')
            black_checkmate = self.is_checkmate(board, 'b')
            stalemate = self.is_stalemate(board)
            threefold = self.is_threefold_repetition(board)
            insufficient = self.is_material_insufficient(board)
            
            return (white_checkmate or black_checkmate or stalemate or 
                    threefold or insufficient)
        except Exception as e:
            return False
    
    def is_checkmate(self, board, side):
        """Check if the given side is in checkmate."""
        if not self.determine_if_checked(board, side):
            return False
            
        # Try all possible moves to see if any get out of check
        for move in self.get_all_legal_moves(side, board):
            new_board = self.make_move(board, move)
            if not self.determine_if_checked(new_board, side):
                return False
        return True
    
    def is_stalemate(self, board):
        """Check if the game is in a stalemate."""
        side = 'w' if self.white_to_move else 'b'
        
        # First check if the side is in check
        if self.determine_if_checked(board, side):
            return False
            
        # Get all legal moves
        legal_moves = self.get_all_legal_moves(side, board)
        
        # Try each possible move
        for move in legal_moves:
            if self.is_move_valid(move, board):
                new_board = self.make_move(board, move)
                if new_board is not None and not self.determine_if_checked(new_board, side):
                    return False
                    
        return True
    
    def get_all_legal_moves(self, side, board=None):
        """Get all legal moves for the given side."""
        if board is None:
            board = self.board
            
        # Check if we're in check first
        check_moves = self.get_check_evasion_moves(side, board)
        if check_moves is not None:  # We're in check
            return check_moves
            
        # Not in check, proceed with normal move generation
        legal_moves = []
        possible_moves = self.get_all_possible_moves_current_pos(side, board)
                
        for piece, positions in possible_moves.items():
            for from_position, to_positions in positions.items():
                for to_position in range(64):
                    if to_positions & (1 << to_position):
                        try:
                            if self.piece_constrainer(from_position=from_position, 
                                                    to_position=to_position, 
                                                    piece=piece, 
                                                    board=board):
                                # Test if this move would leave/put the king in check
                                test_board = self.make_move(board.copy(), (piece, from_position, to_position, None))
                                if test_board and not self.determine_if_checked(test_board, side):
                                    # Check for pawn promotion
                                    if piece[1] == 'P':
                                        if (side == 'w' and to_position >= 56) or (side == 'b' and to_position <= 7):
                                            for promotion_piece in ['Q', 'R', 'B', 'N']:
                                                legal_moves.append((piece, from_position, to_position, promotion_piece))
                                            continue
                                    legal_moves.append((piece, from_position, to_position, None))
                        except Exception as e:
                            continue
                            
        return legal_moves
    
    def is_move_valid(self, move, board):
        """
        Validate a move before attempting to make it.
        
        Args:
            move: Tuple of (piece, from_position, to_position, promotion)
            board: The current board state
            
        Returns:
            bool: True if the move is valid, False otherwise
        """
        piece, from_position, to_position, promotion = move
        
        # Get own and opponent pieces
        side = piece[0]
        own_pieces = 0
        opponent_pieces = 0
        for p in ['P', 'R', 'N', 'B', 'Q', 'K']:
            own_pieces |= board.get(side + p, 0)
            opponent_pieces |= board.get(('b' if side == 'w' else 'w') + p, 0)
            
        # Check if target square is occupied by own piece
        target_bit = 1 << to_position
        if target_bit & own_pieces:
            return False
            
        # For sliding pieces (bishop, rook, queen), check path
        if piece[1] in ['B', 'R', 'Q']:
            # Get all pieces
            all_pieces = own_pieces | opponent_pieces
            # Remove the moving piece from consideration
            all_pieces &= ~(1 << from_position)
            
            # Calculate path based on piece type
            path_bits = 0
            x1, y1 = divmod(from_position, 8)
            x2, y2 = divmod(to_position, 8)
            
            dx = 0 if x1 == x2 else (x2 - x1) // abs(x2 - x1)
            dy = 0 if y1 == y2 else (y2 - y1) // abs(y2 - y1)
            
            x, y = x1 + dx, y1 + dy
            while (x, y) != (x2, y2):
                path_bits |= 1 << (x * 8 + y)
                x, y = x + dx, y + dy
                
            # Check if path is clear
            if path_bits & all_pieces:
                return False
                
        return True

    def make_move(self, board, move):
        """Make a move on the board, handling piece removal and placement."""
        if not self.is_move_valid(move, board):
            return None  # Invalid move
            
        piece, from_position, to_position, promotion = move
        new_board = {k: v for k, v in board.items()}  # Create a deep copy
        
        # Remove piece from original position
        new_board[piece] &= ~(1 << from_position)
        
        # Clear any captured pieces from the target square
        side = piece[0]
        opponent_side = 'b' if side == 'w' else 'w'
        for p in ['P', 'R', 'N', 'B', 'Q', 'K']:
            new_board[opponent_side + p] &= ~(1 << to_position)
        
        # If it's a promotion move
        if promotion:
            # Add the promoted piece
            promoted_piece = piece[0] + promotion
            new_board[promoted_piece] |= (1 << to_position)
        else:
            # Set piece in new position
            new_board[piece] |= (1 << to_position)
        
        return new_board
     
    def get_rook_obstacles(self, from_position, to_position, board=None):
        """
        Check for obstacles in the path of a rook's move.
        
        Args:
            from_position (int): Starting position of the rook
            to_position (int): Target position of the rook
            board (dict): Optional board state to check. If None, uses self.board.
            
        Returns:
            bool: True if there are obstacles in the path, False otherwise
            
        Raises:
            InvalidRookMoveError: If the move is not along a rank or file
        """
        if board is None:
            board = self.board
            
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
        for piece, bitboard in board.items():
            if bitboard & path_without_obstacles:
                return True  # Obstacle found
        
        return False  # No obstacles found
                    
    def apply_castling_constraints(self, side="w", direction="k", board=None):
        """
        Apply castling constraints based on the given side and direction.

        Args:
            side (str): The side for castling, either "w" for white or "b" for black.
            direction (str): The direction for castling, either "k" for king-side or "q" for queen-side.
            board (dict): Optional board state to check. If None, uses self.board.

        Returns:
            bool: True if the castling constraints are satisfied, False otherwise.
        """
        if board is None:
            board = self.board
            
        castle_string_expanded = side + "_castle_" + direction
        castle_status = getattr(self, castle_string_expanded)
        
        if not castle_status:
            return False
            
        if side == "w" and self.determine_if_checked(side="w", board=board):
            return False
            
        if side == "b" and self.determine_if_checked(side="b", board=board):
            return False
            
        if self.is_obstructed_while_castling(side=side, direction=direction, board=board):
            return False
            
        if self.is_checked_while_castling(side=side, direction=direction, board=board):
            return False
            
        return True
                    
    def is_checked_while_castling(self, side="w", direction="k", board=None):
        """
        Check if the king would pass through check while castling.
        
        Args:
            side (str): The side castling ('w' or 'b')
            direction (str): The direction of castling ('k' for kingside, 'q' for queenside)
            board (dict): Optional board state to check. If None, uses self.board.
            
        Returns:
            bool: True if the king would pass through check, False otherwise
        """
        if board is None:
            board = self.board
            
        castle_dict = {
            "wk": ["f1", "g1"],
            "wq": ["d1", "c1"],
            "bk": ["f8", "g8"],
            "bq": ["d8", "c8"]
        }
        for square in castle_dict[side + direction]:
            if self.determine_if_checked_while_castling(king_position=self.string_to_position(square), side=side, board=board):
                return True
        return False
    
    def is_obstructed_while_castling(self, side="w", direction="k", board=None):
        """
        Check if there are any pieces between the king and rook for castling.
        
        Args:
            side (str): The side castling ('w' or 'b')
            direction (str): The direction of castling ('k' for kingside, 'q' for queenside)
            board (dict): Optional board state to check. If None, uses self.board.
            
        Returns:
            bool: True if there are pieces between king and rook, False otherwise
        """
        if board is None:
            board = self.board
            
        castle_dict = {
            "wk": ["f1", "g1"],
            "wq": ["d1", "c1"],
            "bk": ["f8", "g8"],
            "bq": ["d8", "c8"]
        }
        for square in castle_dict[side + direction]:
            if self.get_piece_at_square(board, square):
                return True
        return False
    
    def determine_if_checked_while_castling(self, king_position, side="w", board=None):
        """
        Determine if the king is checked during castling.

        Args:
            king_position (int): The position to check for the king
            side (str): The side castling ('w' or 'b')
            board (dict): Optional board state to check. If None, uses self.board.
            
        Returns:
            bool: True if the king would be in check, False otherwise
        """
        if board is None:
            board = self.board
            
        QR_check = self.checking_queen_or_rook(king_position, side, board)
        BQ_check = self.checking_bishop_or_queen(king_position, side, board)
        N_check = self.checking_knight(king_position, side, board)
        P_check = self.checking_pawn(king_position, side, board)
        
        return bool(QR_check or BQ_check or N_check or P_check)
                    
    def is_threefold_repetition(self, board=None):
        """
        Check if the current position has occurred three times.
        
        Args:
            board (dict): Optional board state to check. If None, uses self.board.
            
        Returns:
            bool: True if the position has occurred three times, False otherwise.
        """
        if board is None:
            board = self.board
        board_hash = self.hash_board_state(board)
        return self.board_states.get(board_hash, 0) >= 3

    def is_material_insufficient(self, board=None):
        """
        Check if there is insufficient material for checkmate.
        
        Args:
            board (dict): Optional board state to check. If None, uses self.board.
            
        Returns:
            bool: True if there is insufficient material, False otherwise.
        """
        if board is None:
            board = self.board
            
        # Count the number of each piece on the board
        piece_counts = {piece: bin(bitboard).count('1') for piece, bitboard in board.items()}
        
        # If any pawns or queens exist, there is sufficient material
        if piece_counts['wP'] > 0 or piece_counts['bP'] > 0 or piece_counts['wQ'] > 0 or piece_counts['bQ'] > 0:
            return False
            
        # If any rooks exist, there is sufficient material
        if piece_counts['wR'] > 0 or piece_counts['bR'] > 0:
            return False
            
        # Count total minor pieces (bishops and knights)
        white_minors = piece_counts['wB'] + piece_counts['wN']
        black_minors = piece_counts['bB'] + piece_counts['bN']
        
        # King vs King
        if white_minors == 0 and black_minors == 0:
            return True
            
        # King + minor piece vs King
        if (white_minors == 1 and black_minors == 0) or (white_minors == 0 and black_minors == 1):
            return True
            
        # King + Bishop vs King + Bishop (same colored squares)
        if piece_counts['wB'] == 1 and piece_counts['bB'] == 1 and white_minors == 1 and black_minors == 1:
            # Get positions of bishops
            white_bishop_pos = None
            black_bishop_pos = None
            for pos in range(64):
                if board['wB'] & (1 << pos):
                    white_bishop_pos = pos
                if board['bB'] & (1 << pos):
                    black_bishop_pos = pos
                    
            # Check if bishops are on same colored squares
            if (white_bishop_pos + white_bishop_pos // 8) % 2 == (black_bishop_pos + black_bishop_pos // 8) % 2:
                return True
                
        return False

    def hash_board_state(self, board=None):
        """
        Create a hash of the current board state.
        
        Args:
            board (dict): Optional board state to hash. If None, uses self.board.
            
        Returns:
            str: A hash string representing the board state.
        """
        if board is None:
            board = self.board
        board_string = ''.join(str(board[piece]) for piece in sorted(board))
        return hashlib.sha256(board_string.encode()).hexdigest()
                    
    def apply_pawn_constraints(self, from_position=0, to_position=0, pawn_type="w", board=None):
        """Check if a pawn move follows the movement constraints."""
        if board is None:
            board = self.board

        # Get the direction and starting rank based on pawn color
        direction = -1 if pawn_type == "w" else 1
        starting_rank = 6 if pawn_type == "w" else 1
        
        # Get current position details
        from_rank = from_position // 8
        from_file = from_position % 8
        to_rank = to_position // 8
        to_file = to_position % 8
        
        # Calculate allowed moves
        forward_one = from_position + (8 * direction)
        forward_two = from_position + (16 * direction)
        
        # Get all pieces
        all_pieces = 0
        for piece, bitboard in board.items():
            all_pieces |= bitboard
        
        # Forward moves
        if to_position == forward_one and not (all_pieces & (1 << to_position)):
            return True
        elif from_rank == starting_rank and to_position == forward_two:
            # Check if both squares are empty for two-square advance
            if not (all_pieces & (1 << forward_one)) and not (all_pieces & (1 << forward_two)):
                return True
        
        # Capture moves
        if abs(to_file - from_file) == 1 and to_rank == from_rank + direction:
            # Regular capture
            target_piece = self.get_piece_at_position(board, to_position)
            if target_piece and target_piece[0] != pawn_type:
                return True
            
            # En passant capture
            if to_position == self.en_passant_target:
                return True
        
        return False
                    
    def is_square_protected(self, board, square, side):
        """
        Check if a square is protected by any piece of the given side using legal moves.
        
        Args:
            board (dict): The board state to check
            square (int): The square position to check
            side (str): The side to check ('w' or 'b')
            
        Returns:
            bool: True if the square is protected, False otherwise
        """
        # Temporarily place an opponent's piece on the square
        opponent_side = 'b' if side == 'w' else 'w'
        temp_piece = opponent_side + 'P'  # Use pawn as temporary piece
        
        # Save the original piece if any
        original_piece = None
        original_bitboard = None
        for piece, bitboard in board.items():
            if bitboard & (1 << square):
                original_piece = piece
                original_bitboard = bitboard
                board[piece] &= ~(1 << square)
                break
        
        # Place temporary piece
        if temp_piece not in board:
            board[temp_piece] = 0
        board[temp_piece] |= (1 << square)
        
        # Check if any piece of our side can legally capture the temporary piece
        is_protected = False
        try:
            # Get all pieces of our side
            for piece_type in ['P', 'N', 'B', 'R', 'Q', 'K']:
                piece = side + piece_type
                piece_bitboard = board.get(piece, 0)
                position = 0
                
                # Check each piece position
                while piece_bitboard:
                    if piece_bitboard & 1:
                        try:
                            # Check if this piece can legally move to the target square
                            if self.piece_constrainer(
                                from_position=position,
                                to_position=square,
                                piece=piece,
                                board=board
                            ):
                                is_protected = True
                                break
                        except Exception:
                            pass
                    piece_bitboard >>= 1
                    position += 1
                    
                if is_protected:
                    break
                    
        except Exception as e:
            print(f"Error checking if square is protected: {str(e)}")
        finally:
            # Restore the original board state
            board[temp_piece] &= ~(1 << square)
            if original_piece:
                board[original_piece] = original_bitboard
        
        return is_protected
                    
    def get_check_evasion_moves(self, side, board=None):
        """Get only moves that can get out of check."""
        if board is None:
            board = self.board
            
        # First check if we're actually in check
        checking_piece_pos = self.determine_if_checked(board, side)
        if not checking_piece_pos:
            return None  # Not in check, use normal move generation
            
        # Get king position
        king_position = self.get_king_position(board, side)
        if king_position is None:
            return []
            
        legal_moves = []
        
        # 1. Get king moves to escape check
        king_piece = side + 'K'
        adj_positions = self.get_adjacent_positions(king_position)
        for to_position in adj_positions:
            if not self.determine_if_king_cant_move_here(board, to_position, side):
                legal_moves.append((king_piece, king_position, to_position, None))
                
        # 2. Try to capture the checking piece
        if checking_piece_pos is not None:
            # Get all pieces that could potentially capture the checking piece
            for piece in ['P', 'N', 'B', 'R', 'Q']:
                piece_bitboard = board.get(side + piece, 0)
                position = 0
                while piece_bitboard:
                    if piece_bitboard & 1:
                        try:
                            if self.piece_constrainer(from_position=position, 
                                                    to_position=checking_piece_pos, 
                                                    piece=side + piece, 
                                                    board=board):
                                # Verify the capture would actually get out of check
                                test_board = self.make_move(board.copy(), (side + piece, position, checking_piece_pos, None))
                                if test_board and not self.determine_if_checked(test_board, side):
                                    legal_moves.append((side + piece, position, checking_piece_pos, None))
                        except Exception:
                            pass
                    piece_bitboard >>= 1
                    position += 1
                    
        # 3. Try to block the check (only for sliding pieces)
        checking_piece = self.get_piece_at_position(board, checking_piece_pos)
        if checking_piece and checking_piece[1] in ['B', 'R', 'Q']:
            # Get squares between checking piece and king
            blocking_squares = self.get_squares_between(checking_piece_pos, king_position)
            
            # Try to find pieces that can block
            for blocking_square in blocking_squares:
                for piece in ['P', 'N', 'B', 'R', 'Q']:
                    piece_bitboard = board.get(side + piece, 0)
                    position = 0
                    while piece_bitboard:
                        if piece_bitboard & 1:
                            try:
                                if self.piece_constrainer(from_position=position, 
                                                        to_position=blocking_square, 
                                                        piece=side + piece, 
                                                        board=board):
                                    # Verify the block would actually get out of check
                                    test_board = self.make_move(board.copy(), (side + piece, position, blocking_square, None))
                                    if test_board and not self.determine_if_checked(test_board, side):
                                        if piece == 'P' and ((side == 'w' and blocking_square >= 56) or (side == 'b' and blocking_square <= 7)):
                                            for promotion_piece in ['Q', 'R', 'B', 'N']:
                                                legal_moves.append((side + piece, position, blocking_square, promotion_piece))
                                        else:
                                            legal_moves.append((side + piece, position, blocking_square, None))
                            except Exception:
                                pass
                        piece_bitboard >>= 1
                        position += 1
                        
        return legal_moves

    def get_squares_between(self, from_pos, to_pos):
        """Get all squares between two positions on a rank, file, or diagonal."""
        squares = []
        
        from_rank, from_file = divmod(from_pos, 8)
        to_rank, to_file = divmod(to_pos, 8)
        
        # Calculate step directions
        rank_step = 0 if from_rank == to_rank else (to_rank - from_rank) // abs(to_rank - from_rank)
        file_step = 0 if from_file == to_file else (to_file - from_file) // abs(to_file - from_file)
        
        # Start from the square after from_pos
        current_rank = from_rank + rank_step
        current_file = from_file + file_step
        
        # Add all squares until we reach to_pos
        while (current_rank, current_file) != (to_rank, to_file):
            if 0 <= current_rank < 8 and 0 <= current_file < 8:
                squares.append(current_rank * 8 + current_file)
            current_rank += rank_step
            current_file += file_step
            
        return squares
                    