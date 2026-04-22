import os
import sys
import time
from pathlib import Path

from sympy import fps

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from kivy.app import App
from kivy.core.window import Window
from kivy.graphics import Color, Line, Rectangle
from kivy.metrics import dp
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.screenmanager import FadeTransition, Screen, ScreenManager
from kivy.uix.widget import Widget

from uttt.game.logic import UltimateTicTacToeGame
from uttt.ml.cnn_agent import build_game_for_mode, load_model


PALETTE = {
    "window_bg": (0.12, 0.12, 0.14, 1.0),
    "panel": (0.18, 0.18, 0.21, 1.0),
    "panel_alt": (0.22, 0.22, 0.26, 1.0),
    "button": (0.28, 0.45, 0.72, 1.0),
    "button_alt": (0.23, 0.58, 0.45, 1.0),
    "board_playable": (0.32, 0.38, 0.32, 1.0),
    "board_blocked": (0.26, 0.26, 0.29, 1.0),
    "board_won": (0.18, 0.18, 0.21, 1.0),
    "grid_line": (0.92, 0.92, 0.95, 1.0),
    "text": (0.96, 0.96, 0.98, 1.0),       #almost white
    "text_muted": (0.80, 0.80, 0.84, 1.0), #grey
    "x": (0.95, 0.78, 0.28, 1.0),
    "o": (0.50, 0.82, 0.95, 1.0),
}


class GradientPane(BoxLayout):
    def __init__(self, **kwargs):
        """build the gradient background pane."""
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(rgba=PALETTE["window_bg"])
            self.bg_rect = Rectangle()
        self.bind(pos=self.update_background, size=self.update_background)
        self.update_background()

    def update_background(self, *_args):
        """resize the background rectangle."""
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size


class SurfaceCard(BoxLayout):
    def __init__(self, fill=None, **kwargs):
        """build a filled panel card."""
        super().__init__(**kwargs)
        self.fill = fill if fill is not None else PALETTE["panel"]

        with self.canvas.before:
            Color(rgba=self.fill)
            self.fill_rect = Rectangle()

        self.bind(pos=self.update_graphics, size=self.update_graphics)
        self.update_graphics()

    def update_graphics(self, *_args):
        """resize the panel background."""
        self.fill_rect.pos = self.pos
        self.fill_rect.size = self.size


class ThemedButton(Button):
    def __init__(self, fill=None, border=None, text_color=None, radius=14, **kwargs):
        """build a themed button."""
        super().__init__(**kwargs)
        self.fill = fill if fill is not None else PALETTE["button"]
        self.background_normal = ""
        self.background_down = ""
        self.background_color = self.fill
        self.color = text_color if text_color is not None else PALETTE["text"]
        self.bold = True


class CellButton(Button):
    def __init__(self, global_r, global_c, **kwargs):
        """build one board cell button."""
        super().__init__(**kwargs)
        self.global_r = global_r
        self.global_c = global_c
        self.font_size = 28
        self.color = PALETTE["text"]
        self.bold = True
        self.background_normal = ""
        self.background_down = ""


class BoardGrid(GridLayout):
    game = ObjectProperty(None)
    status_label = ObjectProperty(None)
    app = ObjectProperty(None)

    def __init__(self, game, status_label, app, **kwargs):
        """build the board grid."""
        super().__init__(**kwargs)
        self.cols = 9
        self.rows = 9
        self.game = game
        self.status_label = status_label
        self.app = app
        self.buttons = []

        for r in range(9):
            row_buttons = []
            for c in range(9):
                btn = CellButton(
                    r,
                    c,
                    text="",
                    background_color=PALETTE["board_blocked"],
                )
                btn.bind(on_release=self.on_cell_pressed)
                row_buttons.append(btn)
                self.add_widget(btn)
            self.buttons.append(row_buttons)

        self.refresh()
        self.bind(size=self.update_grid_lines, pos=self.update_grid_lines)
        self.update_grid_lines()

    def on_cell_pressed(self, btn):
        """handle one player click."""
        if not self.game.is_game_running():
            return

        gr, gc = btn.global_r, btn.global_c
        bi = gr // 3
        r = gr % 3
        bj = gc // 3
        c = gc % 3

        if not self.game.is_valid_move(bi, bj, r, c):
            return

        self.game.apply_move(bi, bj, r, c, self.game.player_symbol)
        self.refresh()

        if not self.game.is_game_running():
            self.show_result()
            return

        self.status_label.text = "AI is thinking..."
        self.status_label.color = PALETTE["text_muted"]

        ai_status = self.app.play_ai_turn()
        self.status_label.text = ai_status
        self.status_label.color = PALETTE["text"]

        self.refresh()
        if not self.game.is_game_running():
            self.show_result()

    def refresh(self):
        """redraw the board state."""
        board = self.game.board_rep
        playable_boards = self.game.get_playable_boards() if self.game.is_game_running() else set()

        for r in range(9):
            for c in range(9):
                bi = r // 3
                bj = c // 3
                btn = self.buttons[r][c]
                sub_status = self.game.sub_boards[bi][bj]

                if sub_status != 0:
                    btn.text = "X" if sub_status == 1 else "O"
                    btn.color = PALETTE["x"] if sub_status == 1 else PALETTE["o"]
                    btn.background_color = PALETTE["board_won"]
                    continue

                value = board[r][c]
                btn.text = "X" if value == 1 else ("O" if value == -1 else "")
                btn.color = PALETTE["x"] if value == 1 else (PALETTE["o"] if value == -1 else PALETTE["text"])

                if (bi, bj) in playable_boards:
                    btn.background_color = PALETTE["board_playable"]
                else:
                    btn.background_color = PALETTE["board_blocked"]

        self.app.update_value_label()

    def show_result(self):
        """show the final result text."""
        winner = self.game.check_true_win()
        if winner == 1:
            self.status_label.text = "X (AI) wins!"
            self.status_label.color = PALETTE["x"]
        elif winner == -1:
            self.status_label.text = "O (YOU) win!"
            self.status_label.color = PALETTE["o"]
        else:
            self.status_label.text = "Tie game."
            self.status_label.color = PALETTE["text_muted"]

    def update_grid_lines(self, *_args):
        """draw the board grid lines."""
        self.canvas.after.clear()
        with self.canvas.after:
            Color(rgba=PALETTE["grid_line"])
            cw = self.width / 9.0
            ch = self.height / 9.0
            x0, y0 = self.x, self.y

            for i in range(10):
                width = 2.4 if i % 3 == 0 else 0.9
                Line(points=[x0 + i * cw, y0, x0 + i * cw, y0 + self.height], width=width)
                Line(points=[x0, y0 + i * ch, x0 + self.width, y0 + i * ch], width=width)


class UTTTApp(App):
    CNN_OPTIONS = ["A", "B", "C", "D", "E"]

    def build(self):
        """build the app root."""
        Window.size = (860, 900)
        Window.clearcolor = PALETTE["window_bg"]

        self.opponent_type = "table"
        self.cnn_option = "C"

        self.screen_manager = ScreenManager(transition=FadeTransition(duration=0.22))
        self.screen_manager.add_widget(self.build_intro_screen())
        self.screen_manager.add_widget(self.build_game_screen())
        return self.screen_manager

    def build_intro_screen(self):
        """build the intro screen."""
        screen = Screen(name="intro")
        root = GradientPane(orientation="vertical", padding=dp(22), spacing=dp(14))

        hero = SurfaceCard(
            orientation="vertical",
            size_hint=(1, 0.26),
            padding=dp(18),
            spacing=dp(8),
            fill=PALETTE["panel_alt"],
        )
        hero.add_widget(
            Label(
                text="[b]Ultimate Tic-Tac-Toe[/b]",
                markup=True,
                color=PALETTE["text"],
                font_size=42,
            )
        )
        hero.add_widget(
            Label(
                text="Pick your opponent and jump into the match.",
                color=PALETTE["text_muted"],
                font_size=19,
            )
        )

        options = SurfaceCard(
            orientation="vertical",
            size_hint=(1, 0.74),
            padding=dp(16),
            spacing=dp(10),
            fill=PALETTE["panel"],
        )
        options.add_widget(
            Label(
                text="[b]Choose Opponent[/b]",
                markup=True,
                color=PALETTE["text"],
                size_hint=(1, 0.12),
                font_size=24,
            )
        )

        quick_row = BoxLayout(size_hint=(1, 0.15), spacing=dp(10))
        qtable_btn = ThemedButton(
            text="Play vs Q-Table",
            font_size=20,
            fill=PALETTE["button_alt"],
        )
        qtable_btn.bind(on_release=lambda x: self.start_game("table"))
        api_btn = ThemedButton(
            text="Play vs API",
            font_size=20,
            fill=PALETTE["button"],
        )
        api_btn.bind(on_release=lambda x: self.start_game("api"))
        quick_row.add_widget(qtable_btn)
        quick_row.add_widget(api_btn)

        options.add_widget(quick_row)
        options.add_widget(
            Label(
                text="Pure CNN choices (5 models). Separate minimax mode uses CNN C.",
                color=PALETTE["text_muted"],
                size_hint=(1, 0.10),
                font_size=18,
            )
        )

        cnn_grid = GridLayout(cols=5, size_hint=(1, 0.40), spacing=dp(8))
        for i, option in enumerate(self.CNN_OPTIONS):
            fill = PALETTE["button"] if i % 2 == 0 else PALETTE["button_alt"]
            btn = ThemedButton(text=f"CNN {option}", font_size=20, fill=fill)
            btn.bind(on_release=lambda _x, m=option: self.start_game("cnn", m))
            cnn_grid.add_widget(btn)
        options.add_widget(cnn_grid)

        minimax_row = BoxLayout(size_hint=(1, 0.10), spacing=dp(10))
        minimax_row.add_widget(Widget())
        minimax_btn = ThemedButton(
            text="Play vs Minimax (CNN C)",
            font_size=20,
            fill=PALETTE["button_alt"],
            size_hint=(0.7, 1),
        )
        minimax_btn.bind(on_release=lambda _x: self.start_game("minimax"))
        minimax_row.add_widget(minimax_btn)
        minimax_row.add_widget(Widget())
        options.add_widget(minimax_row)

        options.add_widget(
            Label(
                text="You can return to this menu any time with the Menu button.",
                color=PALETTE["text_muted"],
                size_hint=(1, 0.13),
                font_size=15,
            )
        )

        root.add_widget(hero)
        root.add_widget(options)
        screen.add_widget(root)
        return screen

    def build_game_screen(self):
        """build the game screen."""
        screen = Screen(name="game")
        root = GradientPane(orientation="vertical", padding=dp(12), spacing=dp(10))

        top = SurfaceCard(
            orientation="horizontal",
            size_hint=(1, 0.11),
            spacing=dp(8),
            padding=dp(8),
            fill=PALETTE["panel_alt"],
        )

        menu_btn = ThemedButton(
            text="Menu",
            size_hint=(0.17, 1),
            font_size=18,
            fill=PALETTE["button"],
        )
        menu_btn.bind(on_release=self.back_to_intro)

        reset_btn = ThemedButton(
            text="Reset",
            size_hint=(0.17, 1),
            font_size=18,
            fill=PALETTE["button_alt"],
        )
        reset_btn.bind(on_release=self.reset_game)

        status_box = BoxLayout(
            orientation="vertical",
            size_hint=(0.66, 1),
            spacing=dp(2),
        )

        self.status_label = Label(
            text="Select mode from intro",
            color=PALETTE["text"],
            size_hint=(1, 0.62),
            font_size=20,
            bold=True,
        )

        self.value_label = Label(
            text="Value: -",
            color=PALETTE["text_muted"],
            size_hint=(1, 0.38),
            font_size=15,
        )
        status_box.add_widget(self.status_label)
        status_box.add_widget(self.value_label)

        top.add_widget(menu_btn)
        top.add_widget(reset_btn)
        top.add_widget(status_box)

        board_shell = SurfaceCard(
            orientation="vertical",
            size_hint=(1, 0.89),
            padding=dp(10),
            fill=PALETTE["panel"],
        )

        temp_game = UltimateTicTacToeGame()
        temp_game.init_game()
        self.board_grid = BoardGrid(temp_game, self.status_label, self, size_hint=(1, 1))
        board_shell.add_widget(self.board_grid)

        root.add_widget(top)
        root.add_widget(board_shell)
        screen.add_widget(root)
        return screen

    def mode_label(self):
        """return the current mode label."""
        if self.opponent_type == "table":
            return "Q-Table"
        if self.opponent_type == "api":
            return "API"
        if self.opponent_type == "minimax":
            return f"Minimax CNN {self.cnn_option}"
        return f"CNN {self.cnn_option}"

    def start_game(self, opponent_type, cnn_option=None):
        """start a new game i n the chosen mode."""
        self.opponent_type = opponent_type
        if opponent_type == "minimax":
            self.cnn_option = "C"
        elif cnn_option is not None:
            self.cnn_option = cnn_option

        self.init_game_instance()
        self.screen_manager.current = "game"
        self.activate_game_board()


    def back_to_intro(self, *_args):
        """return to the intro screen."""
        self.screen_manager.current = "intro"

    def reset_game(self, *_args):
        """reset the current game."""
        if not hasattr(self, "game"):
            return
        self.init_game_instance()
        self.activate_game_board()

    def init_game_instance(self):
        """create the current game object."""
        if self.opponent_type in {"table", "api"}:
            self.game = UltimateTicTacToeGame()
        else:
            if self.opponent_type == "minimax":
                self.cnn_option = "C"
            model, device = load_model(self.cnn_option)
            mode = "minimax_cnn" if self.opponent_type == "minimax" else "pure_cnn"
            self.game = build_game_for_mode(
                model=model,
                device=device,
                mode=mode,
                q_table={},
                training=False,
                multiprocess=False,
            )
        self.game.init_game()

    def activate_game_board(self):
        """bind the current game to the board view."""
        self.board_grid.game = self.game
        self.board_grid.refresh()
        self.status_label.text = f"Mode: {self.mode_label()}"
        self.status_label.color = PALETTE["text"]
        self.update_value_label()

        if not self.game.is_game_running():
            return

        self.play_ai_turn()
        self.board_grid.refresh()

        if self.game.is_game_running():
            self.status_label.text = f"Mode: {self.mode_label()} | AI opened. Your turn."
        else:
            self.board_grid.show_result()

    def update_value_label(self):
        """show the value of the board currently on screen."""
        if not hasattr(self, "game"):
            self.value_label.text = "Value: -"
            return

        value = self.game.current_board_value()
        if value is None:
            self.value_label.text = "Value: -"
        else:
            self.value_label.text = f"Value: {value:.4f}"

    def play_ai_turn(self):
        """let the current ai make one move."""
        if self.opponent_type == "table":
            used_qtable = self.game.agent_smart_move()
            return "Q-table move" if used_qtable else "Random fallback move"

        if self.opponent_type == "cnn":
            self.game.agent_smart_move()
            return f"CNN {self.cnn_option} move"

        if self.opponent_type == "minimax":
            self.game.agent_smart_move()
            return f"Minimax CNN {self.cnn_option} move"

        try:
            from uttt.ui.api import get_ai_move

            move = get_ai_move(self.game, max_tries=2)
            if move is not None:
                gr, gc = move
                bi, r = divmod(gr, 3)
                bj, c = divmod(gc, 3)
                if self.game.is_valid_move(bi, bj, r, c):
                    self.game.apply_move(bi, bj, r, c, self.game.agent_symbol)
                    return "API move"

            self.game.agent_smart_move()
            return "API invalid -> Q-table fallback"
        except Exception:
            self.game.agent_smart_move()
            return "API error -> Q-table fallback"


if __name__ == "__main__":
    UTTTApp().run()
