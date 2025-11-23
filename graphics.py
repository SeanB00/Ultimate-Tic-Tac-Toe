from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.properties import ObjectProperty

from logic import UltimateTicTacToeGame


class CellButton(Button):
    def __init__(self, global_r, global_c, **kwargs):
        super().__init__(**kwargs)
        self.global_r = global_r
        self.global_c = global_c
        self.font_size = 32
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

        for r in range(9):
            row_buttons = []
            for c in range(9):

                # sub-board shading
                bi = r // 3
                bj = c // 3
                if (bi + bj) % 2 == 0:
                    bg = (0.21, 0.21, 0.27, 1)
                else:
                    bg = (0.26, 0.26, 0.32, 1)

                btn = CellButton(r, c, text="", background_color=bg)
                btn.bind(on_release=self.on_cell_pressed)
                row_buttons.append(btn)
                self.add_widget(btn)

            self.buttons.append(row_buttons)

        self.refresh()

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
        for r in range(9):
            for c in range(9):
                v = board[r][c]
                self.buttons[r][c].text = "X" if v == 1 else ("O" if v == -1 else "")

        if not self.game.is_game_running():
            return

        if self.game.curr_board is None:
            self.status_label.text = "Your turn (O): Play in any unfinished sub-board."
        else:
            bi, bj = self.game.curr_board
            self.status_label.text = f"Your turn (O): Must play in sub-board ({bi+1}, {bj+1})."

    def show_result(self):
        w = self.game.check_true_win()
        if w == 1:
            self.status_label.text = "X (AI) WINS!"
        elif w == -1:
            self.status_label.text = "O (YOU) WIN!"
        else:
            self.status_label.text = "It's a TIE!"


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
