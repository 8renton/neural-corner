"""
Pygame fight viewer for Neural Corner.
Plays back a pre-simulated fight exchange by exchange.
"""
import pygame
from simulation.styles import INSIDE, MID, OUTSIDE, DISTANCE_INDEX

# ── Palette ────────────────────────────────────────────────────────────────
BG            = (10, 10, 18)
RING_FLOOR    = (20, 22, 34)
ROPE          = (190, 160, 80)
CORNER_CLR    = (150, 125, 60)
FIGHTER_A_CLR = (210, 55, 55)
FIGHTER_B_CLR = (55, 110, 215)
HIT_CLR       = (255, 215, 50)
TEXT_PRI      = (230, 230, 230)
TEXT_DIM      = (115, 115, 135)
TEXT_GOLD     = (205, 175, 80)
WIN_CLR       = (80, 205, 120)
PANEL_BG      = (16, 16, 26)
SEP_LINE      = (40, 42, 58)
SQ_FILL_A     = (210, 80, 80)
SQ_FILL_B     = (80, 120, 220)
SQ_EMPTY      = (45, 45, 62)

DIST_CLR = {INSIDE: (200, 85, 85), MID: (200, 175, 75), OUTSIDE: (75, 145, 205)}

# ── Layout ─────────────────────────────────────────────────────────────────
W, H          = 980, 660
RING_L        = 60
RING_T        = 155
RING_R        = 920
RING_B        = 470
RING_MID_Y    = (RING_T + RING_B) // 2 + 10
FIGHTER_Y     = RING_MID_Y

# Fighter X positions per distance state
DIST_POS = {
    INSIDE:  (410, 570),
    MID:     (285, 695),   # will be clamped to RING_R below
    OUTSIDE: (145, 835),
}
# Clamp B positions to ring boundary
DIST_POS = {
    INSIDE:  (410, 570),
    MID:     (290, 690),
    OUTSIDE: (150, 830),
}

ACTION_LABEL = {
    "press_forward":     "PRESS FORWARD",
    "maintain_distance": "MAINTAIN DISTANCE",
    "throw_jab":         "THROW JAB",
    "counter_attack":    "COUNTER ATTACK",
    "defensive_shell":   "DEFENSIVE SHELL",
}

# Timing (milliseconds)
T_SHOW_ACTIONS  = 500
T_SHOW_RESULT   = 650
T_ANIMATE       = 350
T_ROUND_INTRO   = 1100
T_ROUND_SUMMARY = 1400
T_FIGHT_END     = 6000


class FightViewer:
    def __init__(self, fight_result, fighter_a, fighter_b):
        self.fight = fight_result
        self.fa = fighter_a
        self.fb = fighter_b

        pygame.init()
        pygame.display.set_caption("Neural Corner")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()

        self._load_fonts()

        # Running display state
        self._round_num   = 0
        self._exch_num    = 0
        self._exch_total  = 0
        self._rounds_a    = 0
        self._rounds_b    = 0
        self._distance    = MID
        self._action_a    = ""
        self._action_b    = ""
        self._event_text  = ""
        self._hit_on_a    = False
        self._hit_on_b    = False
        self._pos_a       = float(DIST_POS[MID][0])
        self._pos_b       = float(DIST_POS[MID][1])

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        if not self._show_intro():
            return

        rounds_a = rounds_b = 0

        for round_result in self.fight.rounds:
            self._round_num  = round_result.round_num
            self._exch_total = round_result.exchanges

            if not self._show_round_intro():
                return

            current_dist = MID
            for i, exchange in enumerate(round_result.exchange_log):
                self._exch_num = i + 1
                if not self._play_exchange(exchange, current_dist, rounds_a, rounds_b):
                    return
                current_dist = exchange.new_distance

            if round_result.winner == "a":
                rounds_a += 1
            elif round_result.winner == "b":
                rounds_b += 1

            self._rounds_a = rounds_a
            self._rounds_b = rounds_b

            if not self._show_round_summary(round_result):
                return

        self._show_fight_end()

    # ------------------------------------------------------------------
    # Scene controllers
    # ------------------------------------------------------------------

    def _show_intro(self) -> bool:
        self.screen.fill(BG)
        a_label = self.font_big.render("NEURAL CORNER", True, TEXT_GOLD)
        vs_label = self.font_title.render(
            f"{self.fa.name}  ({self.fa.style.name})   vs   "
            f"{self.fb.name}  ({self.fb.style.name})",
            True, TEXT_PRI,
        )
        sub = self.font_body.render(
            f"{self.fight.total_rounds}-round fight  ·  Press SPACE to begin", True, TEXT_DIM
        )
        self.screen.blit(a_label, (W // 2 - a_label.get_width() // 2, 200))
        self.screen.blit(vs_label, (W // 2 - vs_label.get_width() // 2, 290))
        self.screen.blit(sub, (W // 2 - sub.get_width() // 2, 360))
        pygame.display.flip()
        return self._wait_for_key_or_time(6000, require_space=True)

    def _show_round_intro(self) -> bool:
        self.screen.fill(BG)
        label = self.font_big.render(f"ROUND  {self._round_num}", True, TEXT_GOLD)
        self.screen.blit(label, (W // 2 - label.get_width() // 2, H // 2 - 30))
        pygame.display.flip()
        return self._wait_for_key_or_time(T_ROUND_INTRO)

    def _show_round_summary(self, round_result) -> bool:
        winner_name = (
            self.fa.name if round_result.winner == "a"
            else self.fb.name if round_result.winner == "b"
            else "DRAW"
        )
        color = (
            FIGHTER_A_CLR if round_result.winner == "a"
            else FIGHTER_B_CLR if round_result.winner == "b"
            else TEXT_GOLD
        )
        self._distance    = MID
        self._action_a    = ""
        self._action_b    = ""
        self._hit_on_a    = False
        self._hit_on_b    = False
        self._event_text  = ""

        self._draw_frame()

        # Overlay
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.screen.blit(overlay, (0, 0))

        label = self.font_big.render(f"Round {self._round_num}  →  {winner_name.upper()}", True, color)
        score = self.font_title.render(
            f"{self.fa.name}  {self._rounds_a}  –  {self._rounds_b}  {self.fb.name}",
            True, TEXT_PRI,
        )
        self.screen.blit(label, (W // 2 - label.get_width() // 2, H // 2 - 50))
        self.screen.blit(score, (W // 2 - score.get_width() // 2, H // 2 + 20))
        pygame.display.flip()
        return self._wait_for_key_or_time(T_ROUND_SUMMARY)

    def _show_fight_end(self) -> None:
        winner_name = (
            self.fa.name if self.fight.winner == "a"
            else self.fb.name if self.fight.winner == "b"
            else "DRAW"
        )
        color = (
            FIGHTER_A_CLR if self.fight.winner == "a"
            else FIGHTER_B_CLR if self.fight.winner == "b"
            else TEXT_GOLD
        )

        self._draw_frame()
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        title  = self.font_big.render("FIGHT RESULT", True, TEXT_GOLD)
        winner = self.font_big.render(f"{winner_name.upper()}  WINS", True, color)
        score  = self.font_title.render(
            f"Rounds:  {self.fa.name} {self.fight.rounds_a}  –  {self.fight.rounds_b} {self.fb.name}"
            + (f"  ({self.fight.draw_rounds} draw)" if self.fight.draw_rounds else ""),
            True, TEXT_PRI,
        )
        sub = self.font_body.render("Press ESC or close window to exit", True, TEXT_DIM)

        self.screen.blit(title,  (W // 2 - title.get_width() // 2,  H // 2 - 110))
        self.screen.blit(winner, (W // 2 - winner.get_width() // 2, H // 2 - 40))
        self.screen.blit(score,  (W // 2 - score.get_width() // 2,  H // 2 + 40))
        self.screen.blit(sub,    (W // 2 - sub.get_width() // 2,    H // 2 + 100))
        pygame.display.flip()
        self._wait_for_key_or_time(T_FIGHT_END)

    # ------------------------------------------------------------------
    # Exchange playback
    # ------------------------------------------------------------------

    def _play_exchange(self, exchange, start_dist: str, rounds_a: int, rounds_b: int) -> bool:
        self._rounds_a   = rounds_a
        self._rounds_b   = rounds_b
        self._distance   = start_dist
        self._action_a   = exchange.action_a
        self._action_b   = exchange.action_b
        self._hit_on_a   = False
        self._hit_on_b   = False
        self._event_text = ""
        self._snap_positions(start_dist)

        # Phase 1: show actions
        self._draw_frame()
        pygame.display.flip()
        if not self._wait_ms(T_SHOW_ACTIONS):
            return False

        # Phase 2: show punch result
        self._hit_on_a   = exchange.punches_b > 0
        self._hit_on_b   = exchange.punches_a > 0
        self._event_text = self._build_event_text(exchange)
        self._draw_frame()
        pygame.display.flip()
        if not self._wait_ms(T_SHOW_RESULT):
            return False

        # Phase 3: animate to new distance if changed
        if exchange.new_distance != start_dist:
            if not self._animate_move(start_dist, exchange.new_distance):
                return False

        self._distance = exchange.new_distance
        return True

    def _animate_move(self, from_dist: str, to_dist: str) -> bool:
        xa0, xb0 = DIST_POS[from_dist]
        xa1, xb1 = DIST_POS[to_dist]
        start_ms = pygame.time.get_ticks()

        while True:
            elapsed = pygame.time.get_ticks() - start_ms
            t = min(1.0, elapsed / T_ANIMATE)
            t_ease = t * t * (3 - 2 * t)   # smoothstep

            self._pos_a = xa0 + (xa1 - xa0) * t_ease
            self._pos_b = xb0 + (xb1 - xb0) * t_ease
            self._distance = to_dist if t > 0.5 else from_dist
            self._hit_on_a = self._hit_on_b = False
            self._event_text = ""

            self._draw_frame(custom_pos=True)
            pygame.display.flip()

            if not self._poll_events():
                return False
            if t >= 1.0:
                break
            self.clock.tick(60)

        self._snap_positions(to_dist)
        return True

    # ------------------------------------------------------------------
    # Core renderer
    # ------------------------------------------------------------------

    def _draw_frame(self, custom_pos: bool = False) -> None:
        self.screen.fill(BG)
        self._draw_top_bar()
        self._draw_score_panel()
        self._draw_ring()

        xa = self._pos_a if custom_pos else float(DIST_POS[self._distance][0])
        xb = self._pos_b if custom_pos else float(DIST_POS[self._distance][1])

        self._draw_fighter(xa, FIGHTER_Y, FIGHTER_A_CLR, self.fa.name.split()[-1], self._hit_on_a, facing_right=True)
        self._draw_fighter(xb, FIGHTER_Y, FIGHTER_B_CLR, self.fb.name.split()[-1], self._hit_on_b, facing_right=False)

        self._draw_action_panel()
        self._draw_distance_bar()
        if self._event_text:
            self._draw_event_banner()

    def _draw_top_bar(self) -> None:
        pygame.draw.line(self.screen, SEP_LINE, (0, 42), (W, 42), 1)
        title = self.font_title.render("NEURAL CORNER", True, TEXT_GOLD)
        self.screen.blit(title, (20, 10))
        if self._round_num:
            rnd = self.font_title.render(
                f"ROUND  {self._round_num} / {self.fight.total_rounds}"
                + (f"   ·   EXCHANGE  {self._exch_num} / {self._exch_total}" if self._exch_num else ""),
                True, TEXT_DIM,
            )
            self.screen.blit(rnd, (W - rnd.get_width() - 20, 10))

    def _draw_score_panel(self) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, (0, 43, W, 112))
        pygame.draw.line(self.screen, SEP_LINE, (0, 155), (W, 155), 1)

        total = self.fight.total_rounds
        sq_size = 16
        sq_gap  = 4

        for side, name, color, sq_fill, rounds_won, x_anchor, align in [
            ("a", self.fa.name, FIGHTER_A_CLR, SQ_FILL_A, self._rounds_a, 30, "left"),
            ("b", self.fb.name, FIGHTER_B_CLR, SQ_FILL_B, self._rounds_b, W - 30, "right"),
        ]:
            label = self.font_title.render(name, True, color)
            style_lbl = self.font_small.render(
                self.fa.style.name if side == "a" else self.fb.style.name, True, TEXT_DIM
            )

            if align == "left":
                self.screen.blit(label, (x_anchor, 52))
                self.screen.blit(style_lbl, (x_anchor, 80))
                sq_x = x_anchor
            else:
                self.screen.blit(label, (x_anchor - label.get_width(), 52))
                self.screen.blit(style_lbl, (x_anchor - style_lbl.get_width(), 80))
                sq_x = x_anchor - total * (sq_size + sq_gap) + sq_gap

            for i in range(total):
                clr = sq_fill if i < rounds_won else SQ_EMPTY
                pygame.draw.rect(self.screen, clr, (sq_x + i * (sq_size + sq_gap), 105, sq_size, sq_size))

        # VS divider
        vs = self.font_body.render("VS", True, TEXT_DIM)
        self.screen.blit(vs, (W // 2 - vs.get_width() // 2, 90))

    def _draw_ring(self) -> None:
        # Floor
        pygame.draw.rect(self.screen, RING_FLOOR, (RING_L, RING_T, RING_R - RING_L, RING_B - RING_T))

        # Ropes (3 horizontal lines at upper third of ring)
        for y_off in [28, 60, 92]:
            pygame.draw.line(self.screen, ROPE,
                             (RING_L, RING_T + y_off), (RING_R, RING_T + y_off), 3)

        # Ring border
        pygame.draw.rect(self.screen, ROPE,
                         (RING_L, RING_T, RING_R - RING_L, RING_B - RING_T), 4)

        # Corner posts
        for cx, cy in [(RING_L, RING_T), (RING_R, RING_T), (RING_L, RING_B), (RING_R, RING_B)]:
            pygame.draw.circle(self.screen, CORNER_CLR, (cx, cy), 9)

    def _draw_fighter(self, x: float, y: int, color, label: str, hit: bool, facing_right: bool) -> None:
        xi = int(x)
        c = HIT_CLR if hit else color

        # Body
        pygame.draw.rect(self.screen, c, (xi - 20, y - 50, 40, 60), border_radius=10)
        # Head
        pygame.draw.circle(self.screen, c, (xi, y - 65), 18)

        # Lead glove (toward opponent)
        gx = xi + 28 if facing_right else xi - 28
        pygame.draw.circle(self.screen, c, (gx, y - 38), 12)

        # Rear glove
        rx = xi - 18 if facing_right else xi + 18
        pygame.draw.circle(self.screen, c, (rx, y - 25), 10)

        # Name tag
        tag = self.font_small.render(label, True, (255, 255, 255))
        self.screen.blit(tag, (xi - tag.get_width() // 2, y + 18))

        # Hit flash ring
        if hit:
            pygame.draw.circle(self.screen, HIT_CLR, (xi, y - 40), 38, 3)

    def _draw_action_panel(self) -> None:
        y_top = RING_B + 12
        pygame.draw.line(self.screen, SEP_LINE, (0, y_top - 2), (W, y_top - 2), 1)

        if self._action_a:
            lbl_a = self.font_action.render(
                f"{self.fa.name}: {ACTION_LABEL.get(self._action_a, self._action_a)}", True, FIGHTER_A_CLR
            )
            self.screen.blit(lbl_a, (30, y_top + 4))

        if self._action_b:
            lbl_b = self.font_action.render(
                f"{self.fb.name}: {ACTION_LABEL.get(self._action_b, self._action_b)}", True, FIGHTER_B_CLR
            )
            self.screen.blit(lbl_b, (W - lbl_b.get_width() - 30, y_top + 4))

    def _draw_distance_bar(self) -> None:
        y = RING_B + 36
        label = self.font_small.render("DISTANCE", True, TEXT_DIM)
        self.screen.blit(label, (30, y))

        seg_w, seg_h = 110, 22
        bar_x = W // 2 - (seg_w * 3) // 2

        for i, dist in enumerate([INSIDE, MID, OUTSIDE]):
            rect = pygame.Rect(bar_x + i * seg_w, y, seg_w, seg_h)
            fill = DIST_CLR[dist] if dist == self._distance else (35, 35, 50)
            pygame.draw.rect(self.screen, fill, rect, border_radius=4)
            pygame.draw.rect(self.screen, (70, 70, 90), rect, 1, border_radius=4)

            txt_clr = TEXT_PRI if dist == self._distance else TEXT_DIM
            txt = self.font_small.render(dist.upper(), True, txt_clr)
            self.screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

    def _draw_event_banner(self) -> None:
        txt = self.font_body.render(self._event_text, True, HIT_CLR)
        self.screen.blit(txt, (W // 2 - txt.get_width() // 2, RING_B + 62))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _build_event_text(self, exchange) -> str:
        if exchange.punches_a > 0 and exchange.punches_b > 0:
            return "BOTH LAND!"
        if exchange.punches_a > 0:
            return f"{self.fa.name} LANDS CLEAN"
        if exchange.punches_b > 0:
            return f"{self.fb.name} LANDS CLEAN"
        return "EXCHANGE — NO PUNCHES THROUGH"

    def _snap_positions(self, dist: str) -> None:
        self._pos_a, self._pos_b = (float(v) for v in DIST_POS[dist])

    def _load_fonts(self) -> None:
        pygame.font.init()
        self.font_big    = pygame.font.SysFont("Arial", 40, bold=True)
        self.font_title  = pygame.font.SysFont("Arial", 21, bold=True)
        self.font_body   = pygame.font.SysFont("Arial", 17)
        self.font_small  = pygame.font.SysFont("Arial", 14)
        self.font_action = pygame.font.SysFont("Arial", 15, bold=True)

    def _poll_events(self) -> bool:
        """Return False if user wants to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def _wait_ms(self, ms: int) -> bool:
        """Block for ms milliseconds, polling quit events. Returns False on quit."""
        end = pygame.time.get_ticks() + ms
        while pygame.time.get_ticks() < end:
            if not self._poll_events():
                return False
            self.clock.tick(60)
        return True

    def _wait_for_key_or_time(self, ms: int, require_space: bool = False) -> bool:
        """Wait for SPACE (or any key if not require_space) or timeout. ESC quits."""
        end = pygame.time.get_ticks() + ms
        while pygame.time.get_ticks() < end:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    if not require_space or event.key == pygame.K_SPACE:
                        return True
            self.clock.tick(60)
        return True
