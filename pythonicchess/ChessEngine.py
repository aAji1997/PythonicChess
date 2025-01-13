from ChessLogic import ConstrainedGameState, InvalidBishopMoveError, InvalidRookMoveError, InvalidQueenMoveError
import warnings
class GameEngine(ConstrainedGameState):
    def __init__(self):
        super().__init__()
        self.max_depth = 50
        
        self.value_map = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 1000}
    
    def set_piece(self, piece, position, board):
        # Determine the color of the piece
        color = 'w' if 'w' in piece else 'b'
        
        # Check if any piece of the same color is already at the position
        for p in board:
            if p.startswith(color) and (board[p] & (1 << position)):
                raise ValueError(f"Position {position} is already occupied by a piece of the same color.")
        # Get the opposite color
        opposite_color = 'b' if color == 'w' else 'w'
        # Check if any piece of the opposite color is at the position, and clear it if it is not a king
        for p in board:
            if p.startswith(opposite_color) and (board[p] & (1 << position)) and not p.endswith('K'):
                self.clear_piece(p, position)
        # Check if the opposite colored king is at the position
        if self.board[opposite_color + 'K'] & (1 << position):
            raise ValueError(f"A king cannot be captured: {opposite_color}K at position {position}")
        
        # Set the piece if the position is not occupied
        board[piece] |= 1 << position

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
        for piece, bitboard in board.items():
            if piece[0] == side and piece[1] == "K":
                position = 0
                while bitboard:
                    if bitboard & 1:
                        return position
                    bitboard >>= 1
                    position += 1

    def determine_if_checked(self, board, side="w"):
        """
        Determine if the specified side's king is checked. 

        Args:
            board (dict): The board state.
            side (str): The side of the board to check for king's position, defaults to "w".

        Returns:
            bool: True if the king is checked, False otherwise.
        """
        #print("Getting king position...")
        #print("Position Board: ", board)
        king_position = self.get_king_position(board, side)
        #print("King position:", self.position_to_string(king_position))
        #print("Checking if king is checked by rook or queen...")
        adj_k = self.are_kings_adjacent(board=board, side=side, to_position=king_position)
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
    
    def are_kings_adjacent(self, board, side, to_position=None):
        """
        Check if the king is moving to a square adjacent to the opponent's king.
    
        Args:
            side (str): The side moving, either 'w' or 'b'.
            board (dict): The board state.
            to_position (int): The bitboard position to move the king to. If None, use the current position of the king.
    
        Returns: 
            bool: True if the kings are adjacent after moving, False otherwise.
        """
        # Get the current position of the side's king if to_position is None
        if to_position is None:
            print(board)
            to_position = self.get_king_position(board, side)
        
        # Get opponent's king position
        opponent_king_pos = self.get_king_position(board, 'b' if side == 'w' else 'w')
        
        # Get adjacent squares to opponent's king
        adj_squares = self.get_adjacent_positions(opponent_king_pos)
        
        # Check if target position is adjacent 
        return to_position in adj_squares

    def determine_if_king_cant_move_here(self, board, new_king_position, side="w"):
        """
        Determine if the king cannot move to the given position by checking for threats from queen, rook, bishop, knight, and pawn. 

        :param new_king_position: The new position of the king on the board.
        :param side: The side of the player, default is "w" for white.
        :return: Returns False if the king is not threatened, otherwise returns the type of piece that is threatening the king.
        """
        # Save current board state
        curr_king_position = self.get_king_position(side)
        old_board = board.copy()
        
        # Temporarily move the king
        self.clear_piece(piece=side+"K", position=curr_king_position)
        self.set_piece(piece=side+"K", position=new_king_position)
        
        # Perform checks
        adj_k = self.are_kings_adjacent(board=board, side=side, to_position=new_king_position)
        if adj_k:
            board = old_board
            return adj_k

        QR_check = self.checking_queen_or_rook(new_king_position, side)
        if QR_check:
            board = old_board
            return QR_check

        BQ_check = self.checking_bishop_or_queen(new_king_position, side)
        if BQ_check:
            board = old_board
            return BQ_check

        N_check = self.checking_knight(new_king_position, side)
        if N_check:
            board = old_board
            return N_check

        P_check = self.checking_pawn(new_king_position, side)
        if P_check:
            board = old_board
            return P_check

        # Restore the board to the previous state once, before returning
        board = old_board
        return False
    def king_can_move(self, board, side="w"):
        """
        Checks if the king can move to any adjacent positions. Returns True if the king can move, and False otherwise.
        Parameters:
            board (dict): The board state
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
            elif self.determine_if_king_cant_move_here(board, position, side):
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
    
    def can_piece_block_or_capture(self, board, target_position, side="w"):
        # Get the position of the side's own king to skip it
        own_king_position = self.get_king_position(board, side)
        
        # Check for any piece that can reach the target position
        for piece, bitboard in board.items():
            if piece.startswith(side) and not piece.endswith('K'):  # Skip the king
                for position in range(64):
                    #print(f"Trying position {self.position_to_string(position)} for {piece}")
                    if bitboard & (1 << position) and position != own_king_position:  # Skip the king's position
                        #print("Present")
                        # Check if the piece can legally move to the target position
                        try:
                            if self.piece_constrainer(from_position=position, to_position=target_position, piece=piece):
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

    def piece_constrainer(self, from_position=-1, to_position=-1, piece="wP"):
        """
        Check if a given piece follows the specified constraints for a move.

        Parameters:
            from_position (int): The starting bitboard position of the piece.
            to_position (int): The ending bitboard position of the piece.
            piece (str): The type of piece being moved. Defaults to "wP".

        Returns:
            bool: True if the piece follows the constraints, False otherwise.
        """
        follows_constraint = False
        if self.white_to_move and piece[0] == "w" or not self.white_to_move and piece[0] == "b":
        
            if piece[1] == "P" or piece == None:
                follows_constraint = self.apply_pawn_constraints(from_position=from_position, to_position=to_position, pawn_type=piece[0])
            
            elif piece[1] == "B":
                follows_constraint = self.apply_bishop_constraints(from_position=from_position, to_position=to_position)

            elif piece[1] == "N":
                follows_constraint = self.apply_knight_constraints(from_position=from_position, to_position=to_position)
            
            elif piece[1] == "R":
                follows_constraint = self.apply_rook_constraints(from_position=from_position, to_position=to_position)
            elif piece[1] == "Q":
                follows_constraint = self.apply_queen_constraints(from_position=from_position, to_position=to_position)
            
            elif piece[1] == "K":
                follows_constraint = self.apply_king_constraints(from_position=from_position, to_position=to_position) or self.apply_castling_constraints(side=piece[0], direction="k") or self.apply_castling_constraints(side=piece[0], direction="q")
                        
            else:
                # log warning saying constraint isn't implemented
                follows_constraint = True
                warnings.warn(f"Constraint not implemented for {piece[1]}")
        else:
            raise ValueError(f"Not the right color to move: {piece}")
            

        return follows_constraint
    
    def minmax(self, board, depth, alpha, beta, maximizing_player):
        """
        Minimax algorithm with alpha-beta pruning to find the best move.
    
        :param board: The current board state.
        :param depth: The maximum depth to search.
        :param alpha: The alpha value for alpha-beta pruning.
        :param beta: The beta value for alpha-beta pruning.
        :param maximizing_player: True if the current player is the maximizing player, False otherwise.
        :return: A tuple containing the best score and the best move.
        """
        if depth == 0 or self.is_game_over(board):
            return self.evaluate_board(board), None
    
        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for move in self.get_all_legal_moves(board, 'w'):
                new_board = self.make_move(board, move)
                eval, _ = self.minmax(new_board, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in self.get_all_legal_moves(board, 'b'):
                new_board = self.make_move(board, move)
                eval, _ = self.minmax(new_board, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move
    
    def is_game_over(self, board):
        """
        Check if the game is over (checkmate or stalemate).
    
        :param board: The current board state.
        :return: True if the game is over, False otherwise.
        """
        return self.is_checkmate(board, 'w') or self.is_checkmate(board, 'b') or self.is_stalemate(board)
    
    def is_checkmate(self, board, side):
        """
        Check if the given side is in checkmate.
    
        :param board: The current board state.
        :param side: The side to check for checkmate ('w' for white, 'b' for black).
        :return: True if the side is in checkmate, False otherwise.
        """
        if not self.determine_if_checked(board, side):
            return False
        for move in self.get_all_legal_moves(board, side):
            new_board = self.make_move(board, move)
            if not self.determine_if_checked(new_board, side):
                return False
        return True
    
    def is_stalemate(self, board):
        """
        Check if the game is in a stalemate.
    
        :param board: The current board state.
        :return: True if the game is in a stalemate, False otherwise.
        """
        side = 'w' if self.white_to_move else 'b'
        if self.determine_if_checked(board, side):
            return False
        for move in self.get_all_legal_moves(board, side):
            new_board = self.make_move(board, move)
            if not self.determine_if_checked(new_board, side):
                return False
        return True
    
    def get_all_legal_moves(self, board, side):
        """
        Get all legal moves for the given side, including pawn promotion options.
    
        :param board: The current board state.
        :param side: The side to get legal moves for ('w' for white, 'b' for black).
        :return: A list of all legal moves, where each move is a tuple of (piece, from_position, to_position, promotion_piece).
                promotion_piece is None for non-promotion moves.
        """
        legal_moves = []
        for piece, positions in self.get_all_possible_moves_current_pos(side).items():
            for from_position, to_positions in positions.items():
                for to_position in range(64):
                    if to_positions & (1 << to_position):
                        try:
                            if self.piece_constrainer(from_position, to_position, piece):
                                # Check for pawn promotion
                                if piece[1] == 'P':
                                    # White pawn reaching 8th rank or black pawn reaching 1st rank
                                    if (side == 'w' and to_position >= 56) or (side == 'b' and to_position <= 7):
                                        # Add a move for each possible promotion piece
                                        for promotion_piece in ['Q', 'R', 'B', 'N']:
                                            legal_moves.append((piece, from_position, to_position, promotion_piece))
                                        continue
                                # For non-promotion moves, add None as promotion piece
                                legal_moves.append((piece, from_position, to_position, None))
                        except Exception:
                            continue
        return legal_moves
    
    def make_move(self, board, move):
        """
        Make a move on the board and return the new board state.
    
        :param board: The current board state.
        :param move: A tuple containing (piece, from_position, to_position, promotion_piece).
                    promotion_piece is None for non-promotion moves.
        :return: The new board state after making the move.
        """
        piece, from_position, to_position, promotion_piece = move
        new_board = board.copy()
        
        # Clear the piece from its original position
        self.clear_piece(piece, from_position, new_board)
        
        # If this is a pawn promotion
        if promotion_piece is not None:
            # Set the promoted piece instead of the pawn
            promoted_piece = piece[0] + promotion_piece  # e.g., 'wP' becomes 'wQ'
            self.set_piece(promoted_piece, to_position, new_board)
        else:
            # Normal move
            self.set_piece(piece, to_position, new_board)
            
        return new_board
         

        
                    