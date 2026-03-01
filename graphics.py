import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from kivy.app import App
from kivy.core.window import Window
from kivy.graphics import Color, Ellipse, Line, Rectangle, RoundedRectangle
from kivy.metrics import dp
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.screenmanager import FadeTransition, Screen, ScreenManager
from kivy.uix.widget import Widget

from CNN import OUT_DIR
from CNN_utils import UltimateTicTacToeCNN, load_model
from logic import UltimateTicTacToeGame


THEME = {
    "bg_a": (0.06, 0.09, 0.16, 1.0),
    "bg_b": (0.10, 0.17, 0.28, 0.70),
    "blob_a": (0.18, 0.37, 0.62, 0.20),
    "blob_b": (0.11, 0.72, 0.64, 0.14),
    "surface": (0.11, 0.14, 0.20, 0.93),
    "surface_alt": (0.09, 0.12, 0.18, 0.94),
    "border": (0.36, 0.47, 0.63, 0.70),
    "text": (0.93, 0.96, 1.0, 1.0),
    "text_muted": (0.76, 0.83, 0.94, 1.0),
    "accent": (0.17, 0.72, 0.61, 1.0),
    "accent_border": (0.34, 0.88, 0.77, 1.0),
    "alt": (0.27, 0.53, 0.90, 1.0),
    "alt_border": (0.47, 0.69, 0.98, 1.0),
}


def _shade(color, factor):
    return (
        max(0.0, min(1.0, color[0] * factor)),
        max(0.0, min(1.0, color[1] * factor)),
        max(0.0, min(1.0, color[2] * factor)),
        color[3],
    )


class GradientPane(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(rgba=THEME["bg_a"])
            self.bg_main = Rectangle()
            Color(rgba=THEME["bg_b"])
            self.bg_overlay = Rectangle()
            Color(rgba=THEME["blob_a"])
            self.blob_a = Ellipse()
            Color(rgba=THEME["blob_b"])
            self.blob_b = Ellipse()
        self.bind(pos=self._update_bg, size=self._update_bg)
        self._update_bg()

    def _update_bg(self, *_args):
        x, y = self.pos
        w, h = self.size
        self.bg_main.pos = (x, y)
        self.bg_main.size = (w, h)
        self.bg_overlay.pos = (x, y)
        self.bg_overlay.size = (w, h)
        self.blob_a.pos = (x - 0.16 * w, y + 0.58 * h)
        self.blob_a.size = (0.68 * w, 0.50 * h)
        self.blob_b.pos = (x + 0.48 * w, y - 0.08 * h)
        self.blob_b.size = (0.58 * w, 0.44 * h)


class SurfaceCard(BoxLayout):
    def __init__(self, fill=None, border=None, radius=18, **kwargs):
        super().__init__(**kwargs)
        self.fill = fill if fill is not None else THEME["surface"]
        self.border = border if border is not None else THEME["border"]
        self.radius = radius

        with self.canvas.before:
            self.fill_color = Color(rgba=self.fill)
            self.fill_rect = RoundedRectangle(radius=[self.radius])
        with self.canvas.after:
            self.border_color = Color(rgba=self.border)
            self.border_line = Line(width=1.2)

        self.bind(pos=self._update_graphics, size=self._update_graphics)
        self._update_graphics()

    def _update_graphics(self, *_args):
        self.fill_rect.pos = self.pos
        self.fill_rect.size = self.size
        self.border_line.rounded_rectangle = (
            self.x,
            self.y,
            self.width,
            self.height,
            self.radius,
        )


class ThemedButton(Button):
    def __init__(self, fill=None, border=None, text_color=None, radius=14, **kwargs):
        super().__init__(**kwargs)
        self.fill = fill if fill is not None else THEME["alt"]
        self.border = border if border is not None else THEME["alt_border"]
        self.radius = radius

        self.background_normal = ""
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)
        self.color = text_color if text_color is not None else THEME["text"]
        self.bold = True

        with self.canvas.before:
            self.fill_color = Color(rgba=self.fill)
            self.fill_rect = RoundedRectangle(radius=[self.radius])
        with self.canvas.after:
            self.border_color = Color(rgba=self.border)
            self.border_line = Line(width=1.2)

        self.bind(pos=self._update_graphics, size=self._update_graphics, state=self._on_state)
        self._update_graphics()

    def _on_state(self, *_args):
        factor = 0.88 if self.state == "down" else 1.0
        self.fill_color.rgba = _shade(self.fill, factor)
        self.border_color.rgba = _shade(self.border, 0.9 if self.state == "down" else 1.0)

    def _update_graphics(self, *_args):
        self.fill_rect.pos = self.pos
        self.fill_rect.size = self.size
        self.border_line.rounded_rectangle = (
            self.x,
            self.y,
            self.width,
            self.height,
            self.radius,
        )


class CellButton(Button):
    def __init__(self, global_r, global_c, **kwargs):
        super().__init__(**kwargs)
        self.global_r = global_r
        self.global_c = global_c
        self.font_size = 28
        self.color = THEME["text"]
        self.bold = True
        self.background_normal = ""
        self.background_down = ""


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

        self.base_light = (0.18, 0.23, 0.31, 1)
        self.base_dark = (0.14, 0.18, 0.25, 1)
        self.playable_tint = (0.16, 0.65, 0.56, 1)
        self.unavailable_tint = (0.05, 0.07, 0.11, 1)
        self.completed_color = (0.08, 0.10, 0.14, 1)
        self.grid_line_color = (0.85, 0.90, 0.98, 0.95)

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

    def on_cell_pressed(self, btn):
        if not self.game.is_game_running():
            return

        gr, gc = btn.global_r, btn.global_c
        bi = gr // 3
        r = gr % 3
        bj = gc // 3
        c = gc % 3

        if self.game.full_board[bi][bj][r][c] != 0:
            return

        if self.game.curr_board is None:
            if (bi, bj) not in self.game.empty_sub_places:
                return
        elif self.game.curr_board != (bi, bj):
            return

        self.apply_player_move(bi, bj, r, c)
        self.refresh()

        if not self.game.is_game_running():
            self.show_result()
            return

        self.status_label.text = "AI is thinking..."
        self.status_label.color = THEME["text_muted"]
        ai_status = self.app.play_ai_turn()
        self.status_label.text = ai_status
        self.status_label.color = THEME["text"]

        self.refresh()
        if not self.game.is_game_running():
            self.show_result()

    def apply_player_move(self, bi, bj, r, c):
        game = self.game
        game.full_board[bi][bj][r][c] = game.player_symbol
        game.place_in_rep(bi, bj, r, c, game.player_symbol)
        game.empty_places[bi][bj].remove((r, c))

        win_status = game.check_win(game.full_board[bi][bj])
        if win_status != 0 or game.tie(game.full_board[bi][bj], game.empty_places[bi][bj]):
            game.sub_boards[bi][bj] = win_status
            if (bi, bj) in game.empty_sub_places:
                game.empty_sub_places.remove((bi, bj))

        next_board = (r, c)
        game.curr_board = None if game.sub_board_is_done(*next_board) else next_board

    def refresh(self):
        board = self.game.global_board()
        playable_boards = self.game.get_playable_boards() if self.game.is_game_running() else set()

        for r in range(9):
            for c in range(9):
                bi = r // 3
                bj = c // 3
                sub_status = self.game.sub_boards[bi][bj]
                btn = self.buttons[r][c]

                if sub_status != 0:
                    btn.text = "X" if sub_status == 1 else "O"
                    btn.color = (0.99, 0.82, 0.44, 1) if sub_status == 1 else (0.52, 0.88, 0.97, 1)
                    btn.background_color = self.completed_color
                    continue

                value = board[r][c]
                btn.text = "X" if value == 1 else ("O" if value == -1 else "")
                btn.color = (0.99, 0.82, 0.44, 1) if value == 1 else ((0.52, 0.88, 0.97, 1) if value == -1 else THEME["text"])

                base = btn.base_color
                if (bi, bj) in playable_boards:
                    color = self._blend(base, self.playable_tint, 0.45)
                else:
                    color = self._blend(base, self.unavailable_tint, 0.34)
                btn.background_color = color

    def _blend(self, base, tint, factor):
        return tuple((1 - factor) * b + factor * t for b, t in zip(base, tint))

    def show_result(self):
        winner = self.game.check_true_win()
        if winner == 1:
            self.status_label.text = "X (AI) wins!"
            self.status_label.color = (0.99, 0.82, 0.44, 1)
        elif winner == -1:
            self.status_label.text = "O (YOU) win!"
            self.status_label.color = (0.52, 0.88, 0.97, 1)
        else:
            self.status_label.text = "Tie game."
            self.status_label.color = THEME["text_muted"]

    def update_grid_lines(self, *_args):
        self.canvas.after.clear()
        with self.canvas.after:
            Color(rgba=self.grid_line_color)
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
        Window.size = (860, 900)
        Window.clearcolor = THEME["bg_a"]

        self.opponent_type = "table"
        self.cnn_option = "A"
        self.model_cache = {}

        self.screen_manager = ScreenManager(transition=FadeTransition(duration=0.22))
        self.screen_manager.add_widget(self._build_intro_screen())
        self.screen_manager.add_widget(self._build_game_screen())
        return self.screen_manager

    def _build_intro_screen(self):
        screen = Screen(name="intro")
        root = GradientPane(orientation="vertical", padding=dp(22), spacing=dp(14))

        hero = SurfaceCard(
            orientation="vertical",
            size_hint=(1, 0.26),
            padding=dp(18),
            spacing=dp(8),
            fill=(0.12, 0.16, 0.24, 0.95),
        )
        hero.add_widget(
            Label(
                text="[b]Ultimate Tic-Tac-Toe[/b]",
                markup=True,
                color=THEME["text"],
                font_size=42,
            )
        )
        hero.add_widget(
            Label(
                text="Pick your opponent and jump into the match.",
                color=THEME["text_muted"],
                font_size=19,
            )
        )

        options = SurfaceCard(
            orientation="vertical",
            size_hint=(1, 0.74),
            padding=dp(16),
            spacing=dp(10),
            fill=THEME["surface"],
        )
        options.add_widget(
            Label(
                text="[b]Choose Opponent[/b]",
                markup=True,
                color=THEME["text"],
                size_hint=(1, 0.12),
                font_size=24,
            )
        )

        quick_row = BoxLayout(size_hint=(1, 0.15), spacing=dp(10))
        qtable_btn = ThemedButton(
            text="Play vs Q-Table",
            font_size=20,
            fill=THEME["accent"],
            border=THEME["accent_border"],
        )
        qtable_btn.bind(on_release=lambda _x: self.start_game("table"))
        api_btn = ThemedButton(
            text="Play vs API",
            font_size=20,
            fill=THEME["alt"],
            border=THEME["alt_border"],
        )
        api_btn.bind(on_release=lambda _x: self.start_game("api"))
        quick_row.add_widget(qtable_btn)
        quick_row.add_widget(api_btn)

        options.add_widget(quick_row)
        options.add_widget(
            Label(
                text="CNN choices (5 models)",
                color=THEME["text_muted"],
                size_hint=(1, 0.10),
                font_size=18,
            )
        )

        cnn_grid = GridLayout(cols=3, size_hint=(1, 0.50), spacing=dp(8))
        for i, option in enumerate(self.CNN_OPTIONS):
            fill = (0.22, 0.55, 0.92, 1) if i % 2 == 0 else (0.15, 0.62, 0.76, 1)
            border = (0.44, 0.73, 0.99, 1) if i % 2 == 0 else (0.34, 0.84, 0.95, 1)
            btn = ThemedButton(text=f"CNN {option}", font_size=20, fill=fill, border=border)
            btn.bind(on_release=lambda _x, m=option: self.start_game("cnn", m))
            cnn_grid.add_widget(btn)
        cnn_grid.add_widget(Widget())
        options.add_widget(cnn_grid)

        options.add_widget(
            Label(
                text="You can return to this menu any time with the Menu button.",
                color=THEME["text_muted"],
                size_hint=(1, 0.13),
                font_size=15,
            )
        )

        root.add_widget(hero)
        root.add_widget(options)
        screen.add_widget(root)
        return screen

    def _build_game_screen(self):
        screen = Screen(name="game")
        root = GradientPane(orientation="vertical", padding=dp(12), spacing=dp(10))

        top = SurfaceCard(
            orientation="horizontal",
            size_hint=(1, 0.11),
            spacing=dp(8),
            padding=dp(8),
            fill=(0.12, 0.16, 0.24, 0.95),
        )

        menu_btn = ThemedButton(
            text="Menu",
            size_hint=(0.17, 1),
            font_size=18,
            fill=(0.25, 0.45, 0.85, 1),
            border=(0.45, 0.63, 0.96, 1),
        )
        menu_btn.bind(on_release=self.back_to_intro)

        reset_btn = ThemedButton(
            text="Reset",
            size_hint=(0.17, 1),
            font_size=18,
            fill=(0.16, 0.65, 0.56, 1),
            border=(0.34, 0.84, 0.75, 1),
        )
        reset_btn.bind(on_release=self.reset_game)

        self.status_label = Label(
            text="Select mode from intro",
            color=THEME["text"],
            size_hint=(0.66, 1),
            font_size=20,
            bold=True,
        )

        top.add_widget(menu_btn)
        top.add_widget(reset_btn)
        top.add_widget(self.status_label)

        board_shell = SurfaceCard(
            orientation="vertical",
            size_hint=(1, 0.89),
            padding=dp(10),
            fill=THEME["surface_alt"],
        )

        temp_game = UltimateTicTacToeGame()
        temp_game.init_game()
        self.board_grid = BoardGrid(temp_game, self.status_label, self, size_hint=(1, 1))
        board_shell.add_widget(self.board_grid)

        root.add_widget(top)
        root.add_widget(board_shell)
        screen.add_widget(root)
        return screen

    def _mode_label(self):
        if self.opponent_type == "table":
            return "Q-Table"
        if self.opponent_type == "api":
            return "API"
        return f"CNN {self.cnn_option}"

    def start_game(self, opponent_type, cnn_option=None):
        self.opponent_type = opponent_type
        if cnn_option is not None:
            self.cnn_option = cnn_option

        self.init_game_instance()
        self.board_grid.game = self.game
        self.board_grid.refresh()
        self.status_label.text = f"Mode: {self._mode_label()}"
        self.status_label.color = THEME["text"]
        self.screen_manager.current = "game"

    def back_to_intro(self, *_args):
        self.screen_manager.current = "intro"

    def reset_game(self, *_args):
        if not hasattr(self, "game"):
            return
        self.init_game_instance()
        self.board_grid.game = self.game
        self.board_grid.refresh()
        self.status_label.text = f"Mode: {self._mode_label()}"
        self.status_label.color = THEME["text"]

    def init_game_instance(self):
        if self.opponent_type in {"table", "api"}:
            self.game = UltimateTicTacToeGame()
        else:
            cached = self.model_cache.get(self.cnn_option)
            if cached is None:
                model, device = load_model(OUT_DIR, self.cnn_option)
                self.model_cache[self.cnn_option] = (model, device)
            model, device = self.model_cache[self.cnn_option]

            self.game = UltimateTicTacToeCNN(
                model=model,
                device=device,
                mode="meta_only",
                q_table={},
                training=False,
                multiprocess=False,
            )
        self.game.init_game()

    def play_ai_turn(self):
        if self.opponent_type == "table":
            used_qtable = self.game.agent_smart_move()
            return "Q-table move" if used_qtable else "Random fallback move"

        if self.opponent_type == "cnn":
            self.game.agent_smart_move()
            return f"CNN {self.cnn_option} move"

        try:
            from api import get_ai_move

            move = get_ai_move(self.game, max_tries=2)
            if move is not None:
                gr, gc = move
                bi, r = divmod(gr, 3)
                bj, c = divmod(gc, 3)
                if (bi, bj, r, c) in self.game.get_available_moves():
                    self.game.apply_agent_move(bi, bj, r, c)
                    return "API move"

            self.game.agent_smart_move()
            return "API invalid -> Q-table fallback"
        except Exception as exc:
            self.game.agent_smart_move()
            return f"API error -> Q-table fallback"


if __name__ == "__main__":
    UTTTApp().run()
