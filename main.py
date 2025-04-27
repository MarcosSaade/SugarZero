import pygame
import sys
from game import Game

if __name__ == "__main__":
    game = Game(ai_enabled=True)
    game.run()