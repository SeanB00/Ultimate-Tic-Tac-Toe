import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.properties import ObjectProperty
from kivy.graphics import Color, Line

import torch

from logic import UltimateTicTacToeGame
from CNN_utils import UltimateTicTacToeCNN, load_model
from CNN import OUT_DIR


# =====================================================
# CELL BUTTON
# =====================================================
class CellButton(Button):
    def __init__(self, global_r, global_c, **kwargs):
        super().__init__(**kwargs)
        self.global_r = global_r
        self.global_c = global_c
        self.font_size = 32
        self.color = (0.93, 0.96, 1, 1)
        self.background_normal = ''


# =====================================================
# BOARD GRID
# =====================================================
class BoardGrid(GridLayout):
    game = ObjectProperty(None)
    status_label = ObjectProperty(None)
    app = ObjectProperty(None)

    def __init__(self, game, status_label, app, **kwargs):
        super().__init__(**kwargs)
        self.cols = 9
        self.rows = 9
        self.game = game
        self.status_label = status_label
        self.app = app
        self.buttons = []

        self.base_light = (0.24, 0.28, 0.36, 1)
        self.base_dark = (0.16, 0.18, 0.25, 1)
        self.playable_tint = (0.29, 0.63, 0.85, 1)
        self.unavailable_tint = (0.04, 0.05, 0.07, 1)
        self.completed_color = (0.09, 0.10, 0.12, 1)
        self.grid_line_color = (0.85, 0.88, 0.94, 1)

        for r in range(9):
            row_buttons = []
            for c in range(9):

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

    # =====================================================
    # CLICK
    # =====================================================
    def on_cell_pressed(self, btn):

        if not self.game.is_game_running():
            return

        gr, gc = btn.global_r, btn.global_c

        bi = gr // 3
        r  = gr % 3
        bj = gc // 3
        c  = gc % 3

        if self.game.full_board[bi][bj][r][c] != 0:
            return

        if self.game.curr_board is None:
            if (bi, bj) not in self.game.empty_sub_places:
                return
        else:
            if self.game.curr_board != (bi, bj):
                return

        self.apply_player_move(bi, bj, r, c)
        self.refresh()

        if not self.game.is_game_running():
            self.show_result()
            return

        # ================= AI MOVE =================
        if self.app.mode == "table":
            used_qtable = self.game.agent_smart_move()
            if not used_qtable:
                self.status_label.text = "Random move..."
            else:
                self.status_label.text = "Q-table move"
        else:
            self.game.agent_smart_move()
            self.status_label.text = "CNN move"

        self.refresh()

        if not self.game.is_game_running():
            self.show_result()

    # =====================================================
    # PLAYER MOVE
    # =====================================================
    def apply_player_move(self, bi, bj, r, c):
        g = self.game
        g.full_board[bi][bj][r][c] = g.player_symbol
        g.place_in_rep(bi, bj, r, c, g.player_symbol)

        g.empty_places[bi][bj].remove((r, c))

        win_status = g.check_win(g.full_board[bi][bj])
        if win_status != 0 or g.tie(g.full_board[bi][bj], g.empty_places[bi][bj]):
            g.sub_boards[bi][bj] = win_status
            if (bi, bj) in g.empty_sub_places:
                g.empty_sub_places.remove((bi, bj))

        next_board = (r, c)
        g.curr_board = None if g.sub_board_is_done(*next_board) else next_board

    # =====================================================
    # REFRESH
    # =====================================================
    def refresh(self):

        board = self.game.global_board()
        playable_boards = self.game.get_playable_boards() if self.game.is_game_running() else set()

        for r in range(9):
            for c in range(9):

                bi = r // 3
                bj = c // 3
                sub_status = self.game.sub_boards[bi][bj]

                # Fill won subboard fully
                if sub_status != 0:
                    self.buttons[r][c].text = "X" if sub_status == 1 else "O"
                    self.buttons[r][c].background_color = self.completed_color
                    continue

                v = board[r][c]
                self.buttons[r][c].text = "X" if v == 1 else ("O" if v == -1 else "")

                base = self.buttons[r][c].base_color

                if (bi, bj) in playable_boards:
                    color = self._blend(base, self.playable_tint, 0.55)
                else:
                    color = self._blend(base, self.unavailable_tint, 0.35)

                self.buttons[r][c].background_color = color

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


# =====================================================
# APP
# =====================================================
class UTTTApp(App):

    def build(self):

        Window.size = (720, 820)

        self.mode = "table"

        root = BoxLayout(orientation='vertical', padding=10, spacing=10)

        self.status_label = Label(
            text="Choose mode and start",
            size_hint=(1, 0.1),
            font_size=20
        )

        top = BoxLayout(size_hint=(1, 0.1))

        reset = Button(text="Reset", size_hint=(0.2, 1))
        reset.bind(on_release=self.reset_game)

        cnn_btn = Button(text="Play vs CNN")
        table_btn = Button(text="Play vs Q-Table")

        cnn_btn.bind(on_release=lambda x: self.set_mode("cnn"))
        table_btn.bind(on_release=lambda x: self.set_mode("table"))

        top.add_widget(reset)
        top.add_widget(cnn_btn)
        top.add_widget(table_btn)
        top.add_widget(self.status_label)

        root.add_widget(top)

        self.init_game_instance()

        self.board_grid = BoardGrid(self.game, self.status_label, self, size_hint=(1, 0.9))
        root.add_widget(self.board_grid)

        return root

    # =====================================================
    # CREATE GAME OBJECT BASED ON MODE
    # =====================================================
    def init_game_instance(self):

        if self.mode == "table":
            self.game = UltimateTicTacToeGame()
        else:
            model_option = "A"
            run_dir = OUT_DIR
            model, device = load_model(run_dir, model_option)

            self.game = UltimateTicTacToeCNN(
                model=model,
                device=device,
                mode="meta_only",  #meta_only, pure_cnn, local_priority, heuristic
                q_table={},
                training=False,
                multiprocess=False,
            )

        self.game.init_game()

    def set_mode(self, mode):
        self.mode = mode
        self.reset_game()

    def reset_game(self, *args):
        self.init_game_instance()
        self.board_grid.game = self.game
        self.status_label.text = f"Mode: {self.mode}"
        self.board_grid.refresh()


if __name__ == "__main__":
    UTTTApp().run()
