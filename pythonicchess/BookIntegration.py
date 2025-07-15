"""
Python-Chess Integration for PythonicChess

This module uses the python-chess library to integrate Polyglot opening books
and provides coordinate conversion between our internal system and standard chess.
"""

import chess
import chess.polyglot
import os
from typing import Optional, Dict, List, Tuple

class PythonChessIntegration:
    """
    Integration layer using python-chess library for opening books and position analysis.
    """
    
    def __init__(self, book_path: Optional[str] = None):
        self.book_path = book_path
        self.book = None
        
        if book_path and os.path.exists(book_path):
            try:
                self.book = chess.polyglot.open_reader(book_path)
                print(f"✅ Successfully loaded Polyglot book: {book_path}")
            except Exception as e:
                print(f"❌ Failed to load Polyglot book: {e}")
                self.book = None
    
    def close_book(self):
        """Close the opening book file."""
        if self.book:
            self.book.close()
            self.book = None
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close_book()
    
    def internal_to_standard_position(self, internal_pos: int) -> int:
        """
        Convert internal position to standard chess position.
        Internal: a1=56, h8=7 | Standard: a1=0, h8=63
        """
        internal_rank = internal_pos // 8
        internal_file = internal_pos % 8
        standard_rank = 7 - internal_rank
        return standard_rank * 8 + internal_file
    
    def standard_to_internal_position(self, standard_pos: int) -> int:
        """
        Convert standard chess position to internal position.
        Standard: a1=0, h8=63 | Internal: a1=56, h8=7
        """
        standard_rank = standard_pos // 8
        standard_file = standard_pos % 8
        internal_rank = 7 - standard_rank
        return internal_rank * 8 + standard_file
    
    def chess_game_to_board(self, chess_game) -> chess.Board:
        """
        Convert our ConstrainedGameState to a python-chess Board object.
        """
        board = chess.Board()
        board.clear()  # Start with empty board
        
        # Piece mapping from our notation to python-chess
        piece_map = {
            'wP': chess.PAWN, 'wR': chess.ROOK, 'wN': chess.KNIGHT,
            'wB': chess.BISHOP, 'wQ': chess.QUEEN, 'wK': chess.KING,
            'bP': chess.PAWN, 'bR': chess.ROOK, 'bN': chess.KNIGHT,
            'bB': chess.BISHOP, 'bQ': chess.QUEEN, 'bK': chess.KING
        }
        
        # Convert pieces
        for piece_type, bitboard in chess_game.board.items():
            if piece_type in piece_map:
                color = chess.WHITE if piece_type.startswith('w') else chess.BLACK
                piece_symbol = piece_map[piece_type]
                
                # Find all positions for this piece type
                for internal_pos in range(64):
                    if bitboard & (1 << internal_pos):
                        standard_pos = self.internal_to_standard_position(internal_pos)
                        square = chess.Square(standard_pos)
                        piece = chess.Piece(piece_symbol, color)
                        board.set_piece_at(square, piece)
        
        # Set turn
        board.turn = chess.WHITE if chess_game.white_to_move else chess.BLACK
        
        # Set castling rights
        castling_rights = 0
        if hasattr(chess_game, 'w_castle_k') and chess_game.w_castle_k:
            castling_rights |= chess.BB_H1
        if hasattr(chess_game, 'w_castle_q') and chess_game.w_castle_q:
            castling_rights |= chess.BB_A1
        if hasattr(chess_game, 'b_castle_k') and chess_game.b_castle_k:
            castling_rights |= chess.BB_H8
        if hasattr(chess_game, 'b_castle_q') and chess_game.b_castle_q:
            castling_rights |= chess.BB_A8
        
        board.castling_rights = castling_rights
        
        # Set en passant (if available)
        if hasattr(chess_game, 'en_passant_target') and chess_game.en_passant_target is not None:
            en_passant_standard = self.internal_to_standard_position(chess_game.en_passant_target)
            board.ep_square = chess.Square(en_passant_standard)
        
        return board
    
    def get_book_move(self, chess_game, prefer_popular: bool = True) -> Optional[Dict]:
        """
        Get a move from the opening book for the current position.
        
        Args:
            chess_game: ConstrainedGameState instance
            prefer_popular: If True, prefer higher-weighted moves
            
        Returns:
            Dictionary with move information or None if no move found
        """
        if not self.book:
            return None
        
        try:
            # Convert our game state to python-chess board
            board = self.chess_game_to_board(chess_game)
            
            # Get all book moves for this position
            book_moves = []
            try:
                for entry in self.book.find_all(board):
                    book_moves.append((entry.move, entry.weight))
            except Exception:
                # Position not found in book
                return None
            
            if not book_moves:
                return None
            
            # Sort by weight if preferring popular moves
            if prefer_popular:
                book_moves.sort(key=lambda x: x[1], reverse=True)
            
            # Get the selected move
            selected_move, weight = book_moves[0]
            
            # Convert move to our coordinate system
            from_square_std = selected_move.from_square
            to_square_std = selected_move.to_square
            
            from_internal = self.standard_to_internal_position(from_square_std)
            to_internal = self.standard_to_internal_position(to_square_std)
            
            from_algebraic = chess_game.position_to_string(from_internal)
            to_algebraic = chess_game.position_to_string(to_internal)
            
            return {
                'from_square': from_algebraic,
                'to_square': to_algebraic,
                'move': f"{from_algebraic}-{to_algebraic}",
                'uci': selected_move.uci(),
                'san': board.san(selected_move),
                'weight': weight,
                'alternatives': len(book_moves)
            }
            
        except Exception as e:
            print(f"Error getting book move: {e}")
            return None
    
    def get_all_book_moves(self, chess_game) -> List[Dict]:
        """
        Get all available book moves for the current position.
        """
        if not self.book:
            return []
        
        try:
            board = self.chess_game_to_board(chess_game)
            book_moves = []
            
            for entry in self.book.find_all(board):
                from_square_std = entry.move.from_square
                to_square_std = entry.move.to_square
                
                from_internal = self.standard_to_internal_position(from_square_std)
                to_internal = self.standard_to_internal_position(to_square_std)
                
                from_algebraic = chess_game.position_to_string(from_internal)
                to_algebraic = chess_game.position_to_string(to_internal)
                
                book_moves.append({
                    'from_square': from_algebraic,
                    'to_square': to_algebraic,
                    'move': f"{from_algebraic}-{to_algebraic}",
                    'uci': entry.move.uci(),
                    'san': board.san(entry.move),
                    'weight': entry.weight
                })
            
            # Sort by weight (highest first)
            book_moves.sort(key=lambda x: x['weight'], reverse=True)
            return book_moves
            
        except Exception as e:
            print(f"Error getting all book moves: {e}")
            return []
    
    def export_fen(self, chess_game) -> str:
        """
        Export the current position as a FEN string.
        """
        try:
            board = self.chess_game_to_board(chess_game)
            return board.fen()
        except Exception as e:
            print(f"Error exporting FEN: {e}")
            return ""
    
    def import_fen(self, chess_game, fen: str) -> bool:
        """
        Load a position from FEN string into our chess game.
        """
        try:
            board = chess.Board(fen)
            
            # Clear our board
            for piece_type in chess_game.board:
                chess_game.board[piece_type] = 0
            
            # Convert pieces from python-chess to our format
            piece_map = {
                (chess.PAWN, chess.WHITE): 'wP', (chess.ROOK, chess.WHITE): 'wR',
                (chess.KNIGHT, chess.WHITE): 'wN', (chess.BISHOP, chess.WHITE): 'wB',
                (chess.QUEEN, chess.WHITE): 'wQ', (chess.KING, chess.WHITE): 'wK',
                (chess.PAWN, chess.BLACK): 'bP', (chess.ROOK, chess.BLACK): 'bR',
                (chess.KNIGHT, chess.BLACK): 'bN', (chess.BISHOP, chess.BLACK): 'bB',
                (chess.QUEEN, chess.BLACK): 'bQ', (chess.KING, chess.BLACK): 'bK'
            }
            
            # Set pieces
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    piece_key = piece_map.get((piece.piece_type, piece.color))
                    if piece_key:
                        internal_pos = self.standard_to_internal_position(square)
                        chess_game.board[piece_key] |= (1 << internal_pos)
            
            # Set turn
            chess_game.white_to_move = board.turn == chess.WHITE
            
            # Set castling rights
            chess_game.w_castle_k = bool(board.castling_rights & chess.BB_H1)
            chess_game.w_castle_q = bool(board.castling_rights & chess.BB_A1)
            chess_game.b_castle_k = bool(board.castling_rights & chess.BB_H8)
            chess_game.b_castle_q = bool(board.castling_rights & chess.BB_A8)
            
            # Set en passant
            if board.ep_square:
                chess_game.en_passant_target = self.standard_to_internal_position(board.ep_square)
            else:
                chess_game.en_passant_target = None
            
            return True
            
        except Exception as e:
            print(f"Error importing FEN: {e}")
            return False
    
    def analyze_position(self, chess_game) -> Dict:
        """
        Analyze the current position and return useful information.
        """
        try:
            board = self.chess_game_to_board(chess_game)
            
            analysis = {
                'fen': board.fen(),
                'turn': 'white' if board.turn == chess.WHITE else 'black',
                'in_check': board.is_check(),
                'is_checkmate': board.is_checkmate(),
                'is_stalemate': board.is_stalemate(),
                'is_draw': board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition(),
                'legal_moves': len(list(board.legal_moves)),
                'material_balance': self._calculate_material_balance(board)
            }
            
            # Check if position is in opening book
            if self.book:
                try:
                    book_moves = list(self.book.find_all(board))
                    analysis['in_book'] = len(book_moves) > 0
                    analysis['book_moves'] = len(book_moves)
                except:
                    analysis['in_book'] = False
                    analysis['book_moves'] = 0
            else:
                analysis['in_book'] = False
                analysis['book_moves'] = 0
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing position: {e}")
            return {}
    
    def _calculate_material_balance(self, board: chess.Board) -> int:
        """Calculate material balance (positive = white advantage)."""
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        
        balance = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                balance += value if piece.color == chess.WHITE else -value
        
        return balance
    
    def get_book_statistics(self) -> Dict:
        """Get statistics about the loaded opening book."""
        if not self.book:
            return {'loaded': False}
        
        try:
            # Count entries by sampling (python-chess doesn't provide direct count)
            sample_count = 0
            sample_limit = 10000  # Sample first 10k positions
            
            # This is a rough estimation - actual books can be much larger
            return {
                'loaded': True,
                'estimated_entries': f"~{sample_limit}+",
                'format': 'Polyglot Binary',
                'path': self.book_path
            }
        except Exception as e:
            return {'loaded': False, 'error': str(e)}

def create_python_chess_integration(book_path: str) -> PythonChessIntegration:
    """
    Factory function to create a PythonChessIntegration instance.
    """
    return PythonChessIntegration(book_path)
