"""
This module provides mock objects for pygame to allow testing game.py
without actually initializing pygame or creating a display.
"""

import sys

# Create mock classes for pygame
class MockRect:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.center = (x + width // 2, y + height // 2)
        self.top = y
        self.left = x
        self.topright = (x + width, y)
    
    def __eq__(self, other):
        if isinstance(other, MockRect):
            return (self.x == other.x and 
                    self.y == other.y and 
                    self.width == other.width and 
                    self.height == other.height)
        return False

class MockSurface:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def blit(self, source, dest):
        # Mock blit does nothing
        pass
    
    def fill(self, color):
        # Mock fill does nothing
        pass
    
    def get_rect(self, **kwargs):
        rect = MockRect(0, 0, self.width, self.height)
        for key, value in kwargs.items():
            setattr(rect, key, value)
        return rect

class MockFont:
    def __init__(self, name, size):
        self.name = name
        self.size = size
    
    def render(self, text, antialias, color):
        # Return a mock surface that would be the rendered text
        return MockSurface(len(text) * self.size // 2, self.size)

class MockClock:
    def tick(self, fps):
        # Mock tick does nothing
        pass

class MockEvent:
    def __init__(self, type, **kwargs):
        self.type = type
        for key, value in kwargs.items():
            setattr(self, key, value)

class MockDisplay:
    @staticmethod
    def set_mode(size):
        return MockSurface(size[0], size[1])
    
    @staticmethod
    def set_caption(caption):
        pass
    
    @staticmethod
    def flip():
        pass

class MockDraw:
    @staticmethod
    def rect(surface, color, rect, width=0):
        # Mock draw does nothing
        pass

class MockPygame:
    QUIT = 1
    KEYDOWN = 2
    MOUSEBUTTONDOWN = 3
    
    K_r = 114
    K_q = 113
    K_a = 97
    K_s = 115
    K_d = 100
    
    @staticmethod
    def init():
        pass
    
    @staticmethod
    def quit():
        pass
    
    @staticmethod
    def get_events():
        return []
    
    class font:
        @staticmethod
        def SysFont(name, size):
            return MockFont(name, size)
    
    class time:
        @staticmethod
        def Clock():
            return MockClock()
    
    class mouse:
        @staticmethod
        def get_pos():
            return (0, 0)

# Set up mock pygame attributes
pygame = MockPygame()
pygame.display = MockDisplay()
pygame.draw = MockDraw()
pygame.Rect = MockRect
pygame.Surface = MockSurface

# Add to sys.modules so it can be imported with mock functionality
sys.modules['pygame'] = pygame