    def render_pawn_promotion(self):
        if not self.promoting:
            return

        # Define the promotion pane dimensions and position
        pane_width = self.width // 3
        pane_height = self.height // 4
        pane_x = (self.width - pane_width) // 2
        pane_y = (self.height - pane_height) // 2

        # Define the colors
        pane_color = pg.Color(200, 200, 200)
        border_color = pg.Color(0, 0, 0)

        # Draw the promotion pane
        pg.draw.rect(self.screen, pane_color, (pane_x, pane_y, pane_width, pane_height))
        pg.draw.rect(self.screen, border_color, (pane_x, pane_y, pane_width, pane_height), 3)

        # Load and display the promotion piece options
        pieces = ['Q', 'R', 'B', 'N']
        piece_images = {}
        for piece in pieces:
            color_prefix = 'b' if self.white_to_move else 'w'
            piece_string = f'{color_prefix}{piece}'
            piece_images[piece_string] = pg.transform.scale(
                pg.image.load(f'/home/hal/ssdlink/PythonicChess/pythonicchess/images/{piece_string}.png'),
                (self.square_size, self.square_size)
            )
            piece_x = pane_x + pieces.index(piece) * (pane_width // len(pieces))
            self.screen.blit(piece_images[piece_string], (piece_x, pane_y))

        # Event handling for piece selection
        for e in pg.event.get():
            if e.type == pg.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pg.mouse.get_pos()
                if pane_y < mouse_y < pane_y + pane_height:
                    for i, piece in enumerate(pieces):
                        piece_x = pane_x + i * (pane_width // len(pieces))
                        if piece_x < mouse_x < piece_x + (pane_width // len(pieces)):
                            self.promoted_piece = f'{color_prefix}{piece}'
                            color = "b" if self.white_to_move else "w"
                            self.clear_piece(piece=color+"P", position=self.promoting_from_pos)
                            self.set_piece(piece=self.promoted_piece, position=self.promoting_to_pos)
                            move_array = np.array([self.piece_enum["="+self.promoted_piece], self.promoting_from_pos, self.promoting_to_pos], dtype=np.int8)
                            self.move_log = np.vstack([self.move_log, move_array])
                            self.promoting = False
                            return