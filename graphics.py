from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.properties import ObjectProperty

from kivy.graphics import Color, Line

from logic import UltimateTicTacToeGame


class CellButton(Button):
    def __init__(self, global_r, global_c, **kwargs):
        super().__init__(**kwargs)
        self.global_r = global_r
        self.global_c = global_c
        self.font_size = 32
        self.color = (0.93, 0.96, 1, 1)
        self.background_normal = ''


class BoardGrid(GridLayout):
    game = ObjectProperty(None)
    status_label = ObjectProperty(None)

    def __init__(self, game, status_label, **kwargs):
        super().__init__(**kwargs)
        self.cols = 9
        self.rows = 9
        self.game = game
        self.status_label = status_label
        self.buttons = []

        # Color palette tuned for clarity
        self.base_light = (0.24, 0.28, 0.36, 1)
        self.base_dark = (0.16, 0.18, 0.25, 1)
        self.playable_tint = (0.29, 0.63, 0.85, 1)
        self.unavailable_tint = (0.04, 0.05, 0.07, 1)
        self.completed_color = (0.09, 0.10, 0.12, 1)
        self.grid_line_color = (0.85, 0.88, 0.94, 1)

        for r in range(9):
            row_buttons = []
            for c in range(9):

                # sub-board shading
                bi = r // 3
                bj = c // 3
                base = self.base_light if (bi + bj) % 2 == 0 else self.base_dark

                btn = CellButton(r, c, text="", background_color=base)
                btn.base_color = base
                btn.bind(on_release=self.on_cell_pressed)
                row_buttons.append(btn)
                self.add_widget(btn)

            self.buttons.append(row_buttons)

        self.refresh()
        self.bind(size=self.update_grid_lines, pos=self.update_grid_lines)
        self.update_grid_lines()

    def on_cell_pressed(self, btn):

        if not self.game.is_game_running():
            return

        gr, gc = btn.global_r, btn.global_c

        # FIXED MAPPING
        bi = gr // 3
        r  = gr % 3
        bj = gc // 3
        c  = gc % 3

        # illegal placement checks
        if self.game.full_board[bi][bj][r][c] != 0:
            return

        if self.game.curr_board is None:
            if (bi, bj) not in self.game.empty_sub_places:
                return
        else:
            if self.game.curr_board != (bi, bj):
                return

        # apply human move
        self.apply_player_move(bi, bj, r, c)
        self.refresh()

        if not self.game.is_game_running():
            self.show_result()
            return

        # AI move
        self.game.agent_smart_move()
        self.refresh()

        if not self.game.is_game_running():
            self.show_result()

    def apply_player_move(self, bi, bj, r, c):
        g = self.game
        g.full_board[bi][bj][r][c] = g.player_symbol
        g.place_in_rep(bi,bj,r,c,g.player_symbol)

        g.empty_places[bi][bj].remove((r, c))

        win_status = g.check_win(g.full_board[bi][bj])
        if win_status != 0 or g.tie(g.full_board[bi][bj], g.empty_places[bi][bj]):
            g.sub_boards[bi][bj] = win_status
            if (bi, bj) in g.empty_sub_places:
                g.empty_sub_places.remove((bi, bj))

        next_board = (r, c)
        g.curr_board = None if g.sub_board_is_done(*next_board) else next_board

    def refresh(self):
        board = self.game.global_board()
        playable_boards = self.game.get_playable_boards() if self.game.is_game_running() else set()
        for r in range(9):
            for c in range(9):
                v = board[r][c]
                self.buttons[r][c].text = "X" if v == 1 else ("O" if v == -1 else "")

                # Update coloring to highlight legal boards
                bi = r // 3
                bj = c // 3

                base = self.buttons[r][c].base_color

                if self.game.sub_board_is_done(bi, bj):
                    color = self.completed_color
                elif (bi, bj) in playable_boards:
                    color = self._blend(base, self.playable_tint, 0.55)
                else:
                    color = self._blend(base, self.unavailable_tint, 0.35)

                self.buttons[r][c].background_color = color

        if not self.game.is_game_running():
            return

        if self.game.curr_board is None:
            self.status_label.text = "Your turn (O): Play in any unfinished (highlighted) sub-board."
        else:
            bi, bj = self.game.curr_board
            self.status_label.text = f"Your turn (O): Must play in highlighted sub-board ({bi+1}, {bj+1})."

    def _blend(self, base, tint, factor):
        return tuple((1 - factor) * b + factor * t for b, t in zip(base, tint))

    def show_result(self):
        w = self.game.check_true_win()
        if w == 1:
            self.status_label.text = "X (AI) WINS!"
        elif w == -1:
            self.status_label.text = "O (YOU) WIN!"
        else:
            self.status_label.text = "It's a TIE!"

    def update_grid_lines(self, *args):
        self.canvas.after.clear()
        with self.canvas.after:
            Color(rgba=self.grid_line_color)

            cw = self.width / 9.0
            ch = self.height / 9.0
            x0, y0 = self.x, self.y

            for i in range(10):
                width = 3 if i % 3 == 0 else 1
                Line(points=[x0 + i * cw, y0, x0 + i * cw, y0 + self.height], width=width)
                Line(points=[x0, y0 + i * ch, x0 + self.width, y0 + i * ch], width=width)


class UTTTApp(App):
    def build(self):
        Window.size = (720, 820)

        root = BoxLayout(orientation='vertical', padding=10, spacing=10)

        self.game = UltimateTicTacToeGame()
        self.game.init_game()

        self.status_label = Label(
            text="AI is making the first move...",
            size_hint=(1, 0.1),
            font_size=20
        )

        # AI STARTS
        self.game.agent_smart_move()

        self.board_grid = BoardGrid(self.game, self.status_label, size_hint=(1, 0.9))

        top = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        reset = Button(text="Reset", size_hint=(0.2, 1))
        reset.bind(on_release=self.reset_game)

        top.add_widget(reset)
        top.add_widget(self.status_label)

        root.add_widget(top)
        root.add_widget(self.board_grid)

        return root

    def reset_game(self, *args):
        self.game.init_game()
        self.status_label.text = "AI starts..."
        self.game.agent_smart_move()
        self.board_grid.refresh()


if __name__ == "__main__":
    UTTTApp().run()
