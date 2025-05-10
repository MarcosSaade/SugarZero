# game.py

import pygame
import sys
import threading
from queue import Queue
from typing import Tuple
import numpy as np
import torch

from constants import (
    FPS, WIDTH, HEIGHT, TILE_SIZE, STATUS_BAR_HEIGHT,
    GRAY_1, GRAY_2, BLACK, WHITE, STATUS_BAR_COLOR,
    RED_COLORS, BLUE_COLORS, SIMULATION_COUNTS, EXPLORATION_WEIGHT, INITIAL_BOARD
)
from utils import get_top_piece, get_top_empty, is_forward_move
from game_state import GameState
from policy_net import PolicyNet
from value_net import ValueNet
from mcts import puct_search

class Game:
    def __init__(self, ai_enabled=True, device: str = 'cpu'):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("SugarZero")
        self.clock = pygame.time.Clock()

        # Device for torch
        self.device = device

        # Instantiate neural nets
        self.policy_model = PolicyNet().to(self.device)
        self.value_model  = ValueNet().to(self.device)

        # Load trained weights if available
        try:
            self.policy_model.load_state_dict(torch.load("./checkpoints/policy_model_10000.pt", map_location=self.device))
            self.value_model.load_state_dict(torch.load("./checkpoints/value_model_10000.pt", map_location=self.device))
            print("Loaded trained AI!")
        except FileNotFoundError:
            print("No trained model found, using random weights.")

        self.policy_model.eval()
        self.value_model.eval()

        # Initialize the board as a NumPy array copy
        self.board = INITIAL_BOARD.copy()

        self.tiles = []
        self.pieces = []
        self.turn = 1  # 0 = red, 1 = blue
        self.selected = None
        self.game_over = False
        self.winner = None

        # Track forward movement for the first three moves
        self.move_count = 0
        self.moved_forward = {0: False, 1: False}

        # AI settings
        self.ai_enabled = ai_enabled
        self.ai_player = 0
        self.ai_thinking = False
        self.ai_move = None
        self.ai_thread = None
        self.ai_result_queue = Queue()

        self.ai_difficulty = "Medium"  # Easy, Medium, Hard

        self.generate_board()

    def generate_board(self):
        for row in range(3):
            for col in range(3):
                self.tiles.append(pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE))

    def draw_board(self):
        for index, tile in enumerate(self.tiles):
            color = GRAY_1 if index % 2 == 0 else GRAY_2
            pygame.draw.rect(self.screen, color, tile)
        if self.selected is not None:
            pygame.draw.rect(self.screen, (100, 255, 100), self.tiles[self.selected], 3)
        if self.ai_move and not self.game_over:
            start, end = self.ai_move
            pygame.draw.rect(self.screen, (255, 200, 0), self.tiles[start], 2)
            pygame.draw.rect(self.screen, (255, 200, 0), self.tiles[end], 2)

    def draw_pieces(self):
        self.pieces = []
        for square_idx, square in enumerate(self.board):
            row, col = divmod(square_idx, 3)
            for height, piece_color in enumerate(square):
                if piece_color != -1:
                    left = col * TILE_SIZE + TILE_SIZE * 0.3
                    top = row * TILE_SIZE + TILE_SIZE * 0.7 - height * 15
                    piece_rect = pygame.Rect(left, top, 80, 30)
                    self.pieces.append((piece_rect, piece_color))
                    color = RED_COLORS if piece_color == 0 else BLUE_COLORS
                    pygame.draw.rect(self.screen, color[height % 2], piece_rect)

    def draw_status_bar(self):
        status_bar = pygame.Rect(0, HEIGHT - STATUS_BAR_HEIGHT, WIDTH, STATUS_BAR_HEIGHT)
        pygame.draw.rect(self.screen, STATUS_BAR_COLOR, status_bar)
        font = pygame.font.SysFont(None, 36)
        if self.game_over:
            msg = "Game Over! Red Wins" if self.winner == 0 else "Game Over! Blue Wins"
            text = font.render(msg, True, RED_COLORS[0] if self.winner == 0 else BLUE_COLORS[0])
        else:
            turn_msg = "Red's Turn" if self.turn == 0 else "Blue's Turn"
            if self.ai_enabled and self.ai_player == self.turn:
                turn_msg += " (AI)"
            text = font.render(turn_msg, True, RED_COLORS[0] if self.turn == 0 else BLUE_COLORS[0])
        text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT - STATUS_BAR_HEIGHT // 2))
        self.screen.blit(text, text_rect)

        if self.move_count < 6 and not (self.moved_forward[0] and self.moved_forward[1]):
            small = pygame.font.SysFont(None, 24)
            forward_text = small.render(
                "Each player must move forward at least once in first 3 moves",
                True, WHITE
            )
            rect = forward_text.get_rect(center=(WIDTH // 2, HEIGHT - STATUS_BAR_HEIGHT // 4))
            self.screen.blit(forward_text, rect)

        if self.ai_enabled:
            small = pygame.font.SysFont(None, 24)
            ai_text = small.render(f"AI: {self.ai_difficulty}", True, WHITE)
            self.screen.blit(ai_text, (WIDTH - 10 - ai_text.get_width(), HEIGHT - STATUS_BAR_HEIGHT + 10))
            if self.ai_thinking:
                thinking = small.render("AI is thinking...", True, WHITE)
                self.screen.blit(thinking, (WIDTH - 10 - thinking.get_width(), HEIGHT - STATUS_BAR_HEIGHT + 40))

    def mouse_to_square(self, x, y) -> int:
        if x < 0 or x >= WIDTH or y >= HEIGHT - STATUS_BAR_HEIGHT:
            return -1
        col, row = x // TILE_SIZE, y // TILE_SIZE
        if col not in range(3) or row not in range(3):
            return -1
        return row * 3 + col

    def get_valid_moves(self):
        return self.create_game_state().get_valid_moves()

    def is_valid_move(self, start, end):
        return self.create_game_state().is_valid_move(start, end)

    def make_move(self, start, end, is_simulation=False):
        # If game already over or move invalid, set game over and winner then exit
        if self.game_over or not self.is_valid_move(start, end):
            if not self.game_over:
                self.game_over = True
                self.winner = 1 - self.turn
            return

        top_s = get_top_piece(self.board, start)
        top_e = get_top_empty(self.board, end)
        self.board[end][top_e] = self.board[start][top_s]
        self.board[start][top_s] = -1

        if is_forward_move(self.turn, start, end):
            self.moved_forward[self.turn] = True

        self.move_count += 1
        if not is_simulation and self.turn == self.ai_player and self.ai_enabled:
            self.ai_move = (start, end)
        self.turn = 1 - self.turn

        if self.check_game_over():
            self.game_over = True
            self.winner = 1 - self.turn

    def check_game_over(self):
        return len(self.get_valid_moves()) == 0

    def handle_click(self, pos):
        if self.game_over or (self.turn == self.ai_player and self.ai_enabled):
            return
        sq = self.mouse_to_square(*pos)
        if sq == -1:
            return
        if self.selected is None:
            top = get_top_piece(self.board, sq)
            if top != -1 and self.board[sq][top] == self.turn:
                self.selected = sq
        else:
            if self.is_valid_move(self.selected, sq):
                self.make_move(self.selected, sq)
            self.selected = None

    def create_game_state(self):
        return GameState(
            board=self.board,
            turn=self.turn,
            move_count=self.move_count,
            moved_forward=self.moved_forward
        )

    def mcts_search_thread(self, result_queue):
        state = self.create_game_state()
        sims  = SIMULATION_COUNTS[self.ai_difficulty]
        move = puct_search(
            state,
            sims,
            policy_model=self.policy_model,
            value_model=self.value_model,
            cpuct=EXPLORATION_WEIGHT,
            device=self.device
        )
        result_queue.put(move)

    def ai_turn(self):
        if self.turn == self.ai_player and self.ai_enabled and not self.game_over:
            if not self.ai_thinking:
                self.ai_thinking = True
                self.draw_game_state()
                pygame.display.flip()
                self.ai_thread = threading.Thread(
                    target=self.mcts_search_thread,
                    args=(self.ai_result_queue,)
                )
                self.ai_thread.daemon = True
                self.ai_thread.start()
            if not self.ai_result_queue.empty():
                move = self.ai_result_queue.get()
                self.ai_thinking = False
                if move:
                    self.make_move(*move)

    def handle_keypress(self, key):
        if key == pygame.K_r:
            self.ai_thinking = False
            while not self.ai_result_queue.empty():
                self.ai_result_queue.get()
            self.__init__(ai_enabled=self.ai_enabled, device=self.device)
        elif key == pygame.K_q:
            return False
        elif key == pygame.K_a:
            self.ai_enabled = not self.ai_enabled
        elif key == pygame.K_s:
            self.ai_player = 1 - self.ai_player
        elif key == pygame.K_d:
            diffs = ["Easy", "Medium", "Hard"]
            self.ai_difficulty = diffs[(diffs.index(self.ai_difficulty) + 1) % len(diffs)]
        return True

    def draw_game_state(self):
        self.screen.fill(BLACK)
        self.draw_board()
        self.draw_pieces()
        self.draw_status_bar()

    def run(self):
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(pygame.mouse.get_pos())
                elif ev.type == pygame.KEYDOWN:
                    running = self.handle_keypress(ev.key)
            self.ai_turn()
            self.draw_game_state()
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()
