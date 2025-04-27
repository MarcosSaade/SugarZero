import pygame
import sys
import copy
import threading
from queue import Queue
from typing import Tuple

from constants import (
    FPS, WIDTH, HEIGHT, TILE_SIZE, STATUS_BAR_HEIGHT,
    GRAY_1, GRAY_2, BLACK, WHITE, STATUS_BAR_COLOR,
    RED_COLORS, BLUE_COLORS, SIMULATION_COUNTS, INITIAL_BOARD
)
from utils import get_top_piece, get_top_empty, is_forward_move
from game_state import GameState
from mcts import mcts_search

class Game:
    def __init__(self, ai_enabled=True):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("SugarZero")
        self.clock = pygame.time.Clock()

        # Initialize the board with pre-placed pieces
        self.board = copy.deepcopy(INITIAL_BOARD)

        self.tiles = []
        self.pieces = []
        self.turn = 1  # 0 = red, 1 = blue
        self.selected = None
        self.game_over = False
        self.winner = None

        # Track forward movement for the first three moves
        self.move_count = 0
        self.moved_forward = {0: False, 1: False}  # Track if each player has moved forward

        # AI settings
        self.ai_enabled = ai_enabled
        self.ai_player = 0  # AI plays as red (0)
        self.ai_thinking = False
        self.ai_move = None
        self.ai_thread = None
        self.ai_result_queue = Queue()

        # AI difficulty
        self.ai_difficulty = "Medium"  # "Easy", "Medium", "Hard"

        # Generate the board squares
        self.generate_board()

    def generate_board(self):
        """Generate the board squares."""
        for row in range(3):
            for col in range(3):
                self.tiles.append(pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE))

    def draw_board(self):
        """Draw the game board."""
        for index, tile in enumerate(self.tiles):
            if index % 2 == 0:
                color = GRAY_1
            else:
                color = GRAY_2

            pygame.draw.rect(self.screen, color, tile)

        # Highlight selected tile
        if self.selected is not None:
            pygame.draw.rect(self.screen, (100, 255, 100), self.tiles[self.selected], 3)

        # Highlight AI's move
        if self.ai_move and not self.game_over:
            start, end = self.ai_move
            pygame.draw.rect(self.screen, (255, 200, 0), self.tiles[start], 2)
            pygame.draw.rect(self.screen, (255, 200, 0), self.tiles[end], 2)

    def draw_pieces(self):
        """Draw game pieces on the board."""
        self.pieces = []  # Reset pieces display list

        for square_idx, square in enumerate(self.board):
            # Calculate position based on square index
            row = square_idx // 3
            col = square_idx % 3

            # Draw each piece in the stack
            for height, piece_color in enumerate(square):
                if piece_color != -1:  # If not empty
                    # Position piece based on square and stack height
                    left = (col * TILE_SIZE) + (TILE_SIZE * 0.3)
                    top = (row * TILE_SIZE) + (TILE_SIZE * 0.7) - (height * 15)
                    piece_rect = pygame.Rect(left, top, 80, 30)
                    self.pieces.append((piece_rect, piece_color))

                    # Draw the piece
                    if piece_color == 0:  # Red
                        pygame.draw.rect(self.screen, RED_COLORS[height % 2], piece_rect)
                    else:  # Blue
                        pygame.draw.rect(self.screen, BLUE_COLORS[height % 2], piece_rect)

    def draw_status_bar(self):
        """Draw the status bar with game information."""
        # Draw status bar background
        status_bar_rect = pygame.Rect(0, HEIGHT - STATUS_BAR_HEIGHT, WIDTH, STATUS_BAR_HEIGHT)
        pygame.draw.rect(self.screen, STATUS_BAR_COLOR, status_bar_rect)

        # Draw status text
        font = pygame.font.SysFont(None, 36)

        if self.game_over:
            if self.winner == 0:
                text = font.render("Game Over! Red Wins", True, RED_COLORS[0])
            else:
                text = font.render("Game Over! Blue Wins", True, BLUE_COLORS[0])
        else:
            if self.turn == 0:
                text = font.render("Red's Turn (AI)" if self.ai_enabled and self.ai_player == 0 else "Red's Turn", True,
                                   RED_COLORS[0])
            else:
                text = font.render("Blue's Turn (AI)" if self.ai_enabled and self.ai_player == 1 else "Blue's Turn",
                                   True, BLUE_COLORS[0])

        # Center text in status bar
        text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT - STATUS_BAR_HEIGHT // 2))
        self.screen.blit(text, text_rect)

        # Display forward movement requirement if still in first 3 moves and not fulfilled
        if self.move_count < 6 and not (self.moved_forward[0] and self.moved_forward[1]):
            small_font = pygame.font.SysFont(None, 24)
            forward_text = small_font.render(
                f"Each player must move forward at least once in first 3 moves",
                True, WHITE
            )
            forward_rect = forward_text.get_rect(center=(WIDTH // 2, HEIGHT - STATUS_BAR_HEIGHT // 4))
            self.screen.blit(forward_text, forward_rect)

        # Display AI difficulty
        if self.ai_enabled:
            ai_font = pygame.font.SysFont(None, 24)
            ai_text = ai_font.render(f"AI: {self.ai_difficulty}", True, WHITE)
            ai_rect = ai_text.get_rect(topright=(WIDTH - 10, HEIGHT - STATUS_BAR_HEIGHT + 10))
            self.screen.blit(ai_text, ai_rect)

            # Display AI thinking status
            if self.ai_thinking:
                thinking_text = ai_font.render("AI is thinking...", True, WHITE)
                thinking_rect = thinking_text.get_rect(topright=(WIDTH - 10, HEIGHT - STATUS_BAR_HEIGHT + 40))
                self.screen.blit(thinking_text, thinking_rect)

    def mouse_to_square(self, mouse_x, mouse_y) -> int:
        """Convert mouse coordinates to board square index."""
        if mouse_x < 0 or mouse_x >= WIDTH or mouse_y < 0 or mouse_y >= HEIGHT - STATUS_BAR_HEIGHT:
            return -1

        col = mouse_x // TILE_SIZE
        row = mouse_y // TILE_SIZE

        if col < 0 or col > 2 or row < 0 or row > 2:
            return -1

        return row * 3 + col

    def get_top_piece(self, square_idx):
        """Find the index of the topmost piece in a square."""
        return get_top_piece(self.board, square_idx)

    def get_top_empty(self, square_idx):
        """Find the index of the first empty space in a square."""
        return get_top_empty(self.board, square_idx)

    def is_forward_move(self, start, end):
        """Check if a move is forward for the current player."""
        return is_forward_move(self.turn, start, end)

    def is_valid_move(self, start, end):
        """Check if a move is valid according to game rules."""
        # Create a game state from current board
        game_state = self.create_game_state()
        return game_state.is_valid_move(start, end)

    def get_valid_moves(self):
        """Get all valid moves for the current player."""
        game_state = self.create_game_state()
        return game_state.get_valid_moves()

    def make_move(self, start, end, is_simulation=False):
        """Make a move on the board."""
        top_start = self.get_top_piece(start)
        top_end = self.get_top_empty(end)

        # Move the piece
        self.board[end][top_end] = self.board[start][top_start]
        self.board[start][top_start] = -1

        # Update forward movement tracking
        if self.is_forward_move(start, end):
            self.moved_forward[self.turn] = True

        # Update move count
        self.move_count += 1

        # Save AI's last move to highlight it
        if not is_simulation and self.turn == self.ai_player and self.ai_enabled:
            self.ai_move = (start, end)

        # Switch turns
        self.turn = 1 - self.turn

        # Check for game over
        if self.check_game_over():
            self.game_over = True
            self.winner = 1 - self.turn  # The previous player won

    def check_game_over(self):
        """Check if the game is over."""
        valid_moves = self.get_valid_moves()
        return len(valid_moves) == 0

    def handle_click(self, mouse_pos):
        """Handle user mouse clicks."""
        if self.game_over or (self.turn == self.ai_player and self.ai_enabled):
            return

        square = self.mouse_to_square(mouse_pos[0], mouse_pos[1])
        if square == -1:
            return

        # First click - select a square
        if self.selected is None:
            top_piece = self.get_top_piece(square)
            # Check if there's a piece and it belongs to the current player
            if top_piece != -1 and self.board[square][top_piece] == self.turn:
                self.selected = square
        # Second click - attempt to move
        else:
            if self.is_valid_move(self.selected, square):
                self.make_move(self.selected, square)
            self.selected = None

    def create_game_state(self):
        """Create a game state object from current board state."""
        return GameState(
            board=self.board,
            turn=self.turn,
            move_count=self.move_count,
            moved_forward=self.moved_forward
        )

    def mcts_search_thread(self, result_queue):
        """Execute Monte Carlo Tree Search in a separate thread."""
        # Create game state from current board
        game_state = self.create_game_state()
        
        # Set simulation count based on difficulty
        sim_count = SIMULATION_COUNTS[self.ai_difficulty]
        
        # Run MCTS and put result in queue
        best_move = mcts_search(game_state, self.ai_player, sim_count)
        result_queue.put(best_move)

    def ai_turn(self):
        """Handle AI's turn."""
        if self.turn == self.ai_player and self.ai_enabled and not self.game_over:
            # Start the AI computation in a separate thread if not already running
            if not self.ai_thinking:
                self.ai_thinking = True
                # Force a redraw to show "thinking..." message
                self.draw_game_state()
                pygame.display.flip()

                # Start AI thread
                self.ai_thread = threading.Thread(
                    target=self.mcts_search_thread,
                    args=(self.ai_result_queue,)
                )
                self.ai_thread.daemon = True  # Thread will close when main program exits
                self.ai_thread.start()

            # Check if the AI has finished calculating
            if not self.ai_result_queue.empty():
                best_move = self.ai_result_queue.get()
                self.ai_thinking = False

                # Make the selected move
                if best_move:
                    start, end = best_move
                    self.make_move(start, end)

    def handle_keypress(self, key):
        """Handle keyboard inputs."""
        if key == pygame.K_r:
            # Make sure to clean up any running thread before resetting
            self.ai_thinking = False
            while not self.ai_result_queue.empty():
                self.ai_result_queue.get()
            self.__init__(ai_enabled=self.ai_enabled)  # Reset game
        elif key == pygame.K_q:
            return False  # Quit
        elif key == pygame.K_a:
            # Toggle AI
            self.ai_enabled = not self.ai_enabled
        elif key == pygame.K_s:
            # Switch AI player
            self.ai_player = 1 - self.ai_player
        elif key == pygame.K_d:
            # Cycle AI difficulty
            difficulties = ["Easy", "Medium", "Hard"]
            current_index = difficulties.index(self.ai_difficulty)
            self.ai_difficulty = difficulties[(current_index + 1) % len(difficulties)]
        return True

    def draw_game_state(self):
        """Draw the current game state."""
        self.screen.fill(BLACK)
        self.draw_board()
        self.draw_pieces()
        self.draw_status_bar()

    def run(self):
        """Main game loop."""
        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(pygame.mouse.get_pos())
                elif event.type == pygame.KEYDOWN:
                    running = self.handle_keypress(event.key)

            # AI's turn
            self.ai_turn()

            # Drawing
            self.draw_game_state()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()