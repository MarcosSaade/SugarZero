# game.py
#
# Pygame interface: play Sugar against the trained agent (or another human).
# Controls: R restart, Q quit, A toggle AI, S swap AI side, D cycle difficulty.

import os
import sys
import threading

import pygame
import torch

import sugar
from constants import (FPS, WIDTH, HEIGHT, TILE_SIZE, STATUS_BAR_HEIGHT, GRAY_1, GRAY_2, BLACK,
                       WHITE, STATUS_BAR_COLOR, SELECTED_COLOR, LAST_MOVE_COLOR, RED_COLORS,
                       BLUE_COLORS, SIMULATION_COUNTS, DIFFICULTIES, PERFECT, MODEL_PATH)
from mcts import search
from network import SugarNet, MAX_PLIES

RED, BLUE = 0, 1


def load_agent(path=MODEL_PATH):
    ckpt = torch.load(path, map_location="cpu")
    net = SugarNet(**ckpt["net_cfg"])
    net.load_state_dict(ckpt["net"])
    net.eval()
    return net


class Game:
    def __init__(self, ai_enabled=True):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("SugarZero")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.small_font = pygame.font.SysFont(None, 24)

        self.net = load_agent() if os.path.exists(MODEL_PATH) else None
        if self.net is None:
            print(f"No model at {MODEL_PATH}; AI disabled.")
        self.oracle = None  # loaded on first use of the Perfect difficulty (about 20 s, 1 GB)
        self.ai_enabled = ai_enabled and self.net is not None
        self.ai_player = RED
        self.difficulty = "Medium"
        self.reset()

    def reset(self):
        self.pos = sugar.START
        self.selected = None
        self.last_move = None
        self.result = None  # None while playing, else RED, BLUE or "draw"
        self.search_pos = None  # position the running AI search is for
        self.ai_results = []    # (position, move) pairs appended by search threads

    @property
    def turn(self):
        return self.pos[0] >> 63

    def apply_move(self, move):
        self.pos = sugar.play(self.pos, move)
        self.last_move = move
        if not sugar.legal_moves(self.pos):
            self.result = 1 - self.turn
        elif self.pos[1] >= MAX_PLIES:
            self.result = "draw"

    # ---------- drawing ----------

    def tile_rect(self, square):
        row, col = divmod(square, 3)
        return pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)

    def draw_board(self):
        for sq in range(9):
            pygame.draw.rect(self.screen, GRAY_1 if sq % 2 == 0 else GRAY_2, self.tile_rect(sq))
        if self.last_move:
            for sq in self.last_move:
                pygame.draw.rect(self.screen, LAST_MOVE_COLOR, self.tile_rect(sq), 2)
        if self.selected is not None:
            pygame.draw.rect(self.screen, SELECTED_COLOR, self.tile_rect(self.selected), 3)

    def draw_pieces(self):
        for sq, pieces in enumerate(sugar.stacks(self.pos[0])):
            row, col = divmod(sq, 3)
            for height, colour in enumerate(pieces):
                left = col * TILE_SIZE + TILE_SIZE * 0.3
                top = row * TILE_SIZE + TILE_SIZE * 0.7 - height * 15
                shades = RED_COLORS if colour == RED else BLUE_COLORS
                pygame.draw.rect(self.screen, shades[height % 2], pygame.Rect(left, top, 80, 30))

    def draw_status_bar(self):
        bar_top = HEIGHT - STATUS_BAR_HEIGHT
        pygame.draw.rect(self.screen, STATUS_BAR_COLOR, pygame.Rect(0, bar_top, WIDTH, STATUS_BAR_HEIGHT))
        if self.result == "draw":
            msg, colour = f"Draw ({MAX_PLIES}-move limit)", WHITE
        elif self.result is not None:
            msg = "Game Over! Red Wins" if self.result == RED else "Game Over! Blue Wins"
            colour = RED_COLORS[0] if self.result == RED else BLUE_COLORS[0]
        else:
            msg = "Red's Turn" if self.turn == RED else "Blue's Turn"
            if self.ai_enabled and self.ai_player == self.turn:
                msg += " (AI)"
            colour = RED_COLORS[0] if self.turn == RED else BLUE_COLORS[0]
        text = self.font.render(msg, True, colour)
        self.screen.blit(text, text.get_rect(center=(WIDTH // 2, bar_top + STATUS_BAR_HEIGHT // 2)))

        if self.result is None and sugar.rule_active(self.pos[1], self.pos[2], self.turn):
            hint = self.small_font.render("Each player must move forward at least once in their first 3 moves",
                                          True, WHITE)
            self.screen.blit(hint, hint.get_rect(center=(WIDTH // 2, HEIGHT - STATUS_BAR_HEIGHT // 4)))

        if self.ai_enabled:
            label = self.small_font.render(f"AI: {self.difficulty}", True, WHITE)
            self.screen.blit(label, (WIDTH - 10 - label.get_width(), bar_top + 10))
            if self.search_pos == self.pos:
                busy = "Loading perfect player..." if self.difficulty == PERFECT and self.oracle is None \
                    else "AI is thinking..."
                label = self.small_font.render(busy, True, WHITE)
                self.screen.blit(label, (WIDTH - 10 - label.get_width(), bar_top + 40))

    def draw(self):
        self.screen.fill(BLACK)
        self.draw_board()
        self.draw_pieces()
        self.draw_status_bar()
        pygame.display.flip()

    # ---------- input ----------

    def mouse_to_square(self, x, y):
        if not (0 <= x < WIDTH and 0 <= y < HEIGHT - STATUS_BAR_HEIGHT):
            return None
        return (y // TILE_SIZE) * 3 + x // TILE_SIZE

    def handle_click(self, x, y):
        if self.result is not None or (self.ai_enabled and self.turn == self.ai_player):
            return
        sq = self.mouse_to_square(x, y)
        if sq is None:
            return
        if self.selected is None:
            pieces = sugar.stacks(self.pos[0])[sq]
            if pieces and pieces[-1] == self.turn:
                self.selected = sq
        else:
            if (self.selected, sq) in sugar.legal_moves(self.pos):
                self.apply_move((self.selected, sq))
            self.selected = None

    def handle_keypress(self, key):
        if key == pygame.K_r:
            self.reset()
        elif key == pygame.K_q:
            return False
        elif key == pygame.K_a and self.net is not None:
            self.ai_enabled = not self.ai_enabled
        elif key == pygame.K_s:
            self.ai_player = 1 - self.ai_player
        elif key == pygame.K_d:
            self.difficulty = DIFFICULTIES[(DIFFICULTIES.index(self.difficulty) + 1) % len(DIFFICULTIES)]
        return True

    # ---------- AI ----------

    def run_search(self, pos, difficulty):
        if difficulty == PERFECT:
            if self.oracle is None:
                from oracle import Oracle
                self.oracle = Oracle()
            move = self.oracle.perfect_move(pos)
        else:
            moves, visits, _ = search(self.net, [pos], SIMULATION_COUNTS[difficulty])[0]
            move = moves[int(visits.argmax())]
        self.ai_results.append((pos, move))

    def ai_turn(self):
        # Results are keyed by position, so a search that finishes after a restart is ignored.
        while self.ai_results:
            pos, move = self.ai_results.pop()
            if pos == self.pos and self.result is None:
                self.apply_move(move)
        if not self.ai_enabled or self.result is not None or self.turn != self.ai_player:
            return
        if self.search_pos != self.pos:
            self.search_pos = self.pos
            args = (self.pos, self.difficulty)
            threading.Thread(target=self.run_search, args=args, daemon=True).start()

    def run(self):
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(*ev.pos)
                elif ev.type == pygame.KEYDOWN:
                    running = self.handle_keypress(ev.key)
            self.ai_turn()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()
