from ChessLogic import ConstrainedGameState, InvalidBishopMoveError, InvalidRookMoveError, InvalidQueenMoveError
import warnings
from copy import deepcopy
import hashlib

class GameEngine(ConstrainedGameState):
    def __init__(self):
        super().__init__()
        self.max_depth = 50
        
        self.value_map = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 1000}
    
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
            print(f"Warning: Could not find king position for side {side}")
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
            
        possible_moves = {"wP": {}, "wN": {}, "wB": {}, "wR": {}, "wQ": {}, "wK": {}} if side == "w" else {
            "bP": {}, "bN": {}, "bB": {}, "bR": {}, "bQ": {}, "bK": {}}
            
        # Get own and opponent pieces
        own_pieces = 0
        opponent_pieces = 0
        for piece in ['P', 'R', 'N', 'B', 'Q', 'K']:
            own_pieces |= board.get(side + piece, 0)
            opponent_pieces |= board.get(('b' if side == 'w' else 'w') + piece, 0)
            
        # Get bishop moves
        bishop_bitboard = board.get(side + 'B', 0)
        for position in range(64):
            if bishop_bitboard & (1 << position):
                possible_moves_bitboard = self.B_lookup[position]
                possible_moves_bitboard &= ~own_pieces
                if possible_moves_bitboard:
                    possible_moves[side + 'B'][position] = possible_moves_bitboard
        
        # Get knight moves
        knight_bitboard = board.get(side + 'N', 0)
        for position in range(64):
            if knight_bitboard & (1 << position):
                possible_moves_bitboard = self.N_lookup[position]
                possible_moves_bitboard &= ~own_pieces
                if possible_moves_bitboard:
                    possible_moves[side + 'N'][position] = possible_moves_bitboard
        
        # Get rook moves
        rook_bitboard = board.get(side + 'R', 0)
        for position in range(64):
            if rook_bitboard & (1 << position):
                possible_moves_bitboard = self.R_lookup[position]
                possible_moves_bitboard &= ~own_pieces
                if possible_moves_bitboard:
                    possible_moves[side + 'R'][position] = possible_moves_bitboard
        
        # Get queen moves
        queen_bitboard = board.get(side + 'Q', 0)
        for position in range(64):
            if queen_bitboard & (1 << position):
                possible_moves_bitboard = self.Q_lookup[position]
                possible_moves_bitboard &= ~own_pieces
                if possible_moves_bitboard:
                    possible_moves[side + 'Q'][position] = possible_moves_bitboard
        
        # Get king moves
        king_position = self.get_king_position(board, side)
        if king_position is not None:
            possible_moves_bitboard = 0
            for move in self.get_adjacent_positions(king_position):
                if not self.determine_if_king_cant_move_here(board, move, side):
                    move_bit = 1 << move
                    if not (move_bit & own_pieces):
                        possible_moves_bitboard |= move_bit

            # Check for castling moves
            if self.apply_castling_constraints(side, "k", board):
                if side == "w":
                    possible_moves_bitboard |= (1 << 62)
                else:
                    possible_moves_bitboard |= (1 << 6)
            if self.apply_castling_constraints(side, "q", board):
                if side == "w":
                    possible_moves_bitboard |= (1 << 58)
                else:
                    possible_moves_bitboard |= (1 << 2)

            if possible_moves_bitboard:
                possible_moves[side + 'K'][king_position] = possible_moves_bitboard
        
        # Get pawn moves
        pawn_bitboard = board.get(side + 'P', 0)
        for position in range(64):
            if pawn_bitboard & (1 << position):
                possible_moves_bitboard = self.P_lookup_w[position] if side == 'w' else self.P_lookup_b[position]
                forward_mask = possible_moves_bitboard & ~(own_pieces | opponent_pieces)
                capture_mask = possible_moves_bitboard & opponent_pieces
                possible_moves_bitboard = forward_mask | capture_mask
                if possible_moves_bitboard:
                    possible_moves[side + 'P'][position] = possible_moves_bitboard
                
        return possible_moves
        
    def move_piece(self, piece, from_position, to_position, board):
            self.clear_piece(piece, from_position, board)
            self.set_piece(piece, to_position, board)
            
            return board

        
    def evaluate_board(self, board):
        score = 0
        for piece, bitboard in board.items():
            if piece[0] == 'w':
                score += self.value_map[piece[1]]
            else:
                score -= self.value_map[piece[1]]
        return score

    def piece_constrainer(self, from_position=0, to_position=0, piece="wP", board=None):
        """
        Check if a given piece follows the specified constraints for a move.

        Args:
            from_position (int): The starting position of the piece on the bitboard.
            to_position (int): The ending position of the piece on the bitboard.
            piece (str): The type of piece being moved. Defaults to "wP".
            board (dict): Optional board state to check. If None, uses self.board.

        Returns:
            bool: True if the piece follows the constraints, False otherwise.
        """
        if board is None:
            board = self.board
            
        follows_constraint = False
        # During minmax evaluation, we need to check based on the piece color, not the turn state
        if piece[0] == "w" or piece[0] == "b" or "O" in piece:
            if piece[1] == "P" or piece == None:
                follows_constraint = super().apply_pawn_constraints(from_position=from_position, 
                                                                     to_position=to_position, 
                                                                     pawn_type=piece[0])
            
            elif piece[1] == "B":
                follows_constraint = self.apply_bishop_constraints(board=board,
                                                                    from_position=from_position, 
                                                                    to_position=to_position)

            elif piece[1] == "N":
                follows_constraint = super().apply_knight_constraints(from_position=from_position, 
                                                                       to_position=to_position)
            
            elif piece[1] == "R":
                follows_constraint = super().apply_rook_constraints(from_position=from_position, 
                                                                       to_position=to_position)
            elif piece[1] == "Q":
                follows_constraint = super().apply_queen_constraints(from_position=from_position, 
                                                                       to_position=to_position)
            
            elif piece[1] == "K":
                follows_constraint = super().apply_king_constraints(from_position=from_position, 
                                                                       to_position=to_position)
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
            raise ValueError(f"Invalid piece color: {piece[0]}")
            
        return follows_constraint
    
    def minmax(self, board, depth, alpha, beta, maximizing_player):
        """Minimax algorithm with alpha-beta pruning to find the best move."""
        if not isinstance(board, dict):
            return float('-inf') if maximizing_player else float('inf'), None
            
        if depth == 0 or self.is_game_over(board):
            return self.evaluate_board(board), None
    
        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            try:
                moves = self.get_all_legal_moves('w', board)
            except Exception as e:
                return max_eval, None
                
            for move in moves:
                try:
                    new_board = self.make_move(board, move)
                    eval, _ = self.minmax(new_board, depth - 1, alpha, beta, False)
                    if eval > max_eval:
                        max_eval = eval
                        best_move = move
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                except Exception as e:
                    continue
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            try:
                moves = self.get_all_legal_moves('b', board)
            except Exception as e:
                return min_eval, None
                
            for move in moves:
                try:
                    new_board = self.make_move(board, move)
                    eval, _ = self.minmax(new_board, depth - 1, alpha, beta, True)
                    if eval < min_eval:
                        min_eval = eval
                        best_move = move
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                except Exception as e:
                    continue
            return min_eval, best_move
    
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
                    