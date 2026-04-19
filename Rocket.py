
import pygame
import numpy as np
import random

pygame.init()

# =========================
# Constants
# =========================
WIDTH, HEIGHT = 800, 600
GROUND_HEIGHT = 50

GRAVITY = 0.05
ACTIONS = [0, 1, 2, 3]

# RL Hyperparameters
GAMMA = 0.99
ALPHA = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 64
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

# =========================
# Screen Setup
# =========================
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Advanced Rocket Landing - Research Mode")

background_img = pygame.image.load("surface.jpg")
background_img = pygame.transform.scale(background_img, (WIDTH, HEIGHT))

rocket_img = pygame.image.load("rocket.png")
rocket_img = pygame.transform.scale(rocket_img, (40, 80))


# =========================
# Rocket Class
# =========================
class Rocket:

    def __init__(self):
        self.reset()

    def reset(self):
        self.x = WIDTH // 2
        self.y = 100

        self.vx = 0
        self.vy = 0

        self.angle = 0
        self.angular_velocity = 0

        self.fuel = 100

        self.landed = False
        self.crashed = False

    # =========================
    # UPDATE FUNCTION
    # =========================
    def update(self, action):

        if self.landed or self.crashed:
            return 0

        # Gravity
        self.vy += GRAVITY

        # Wind disturbance
        wind = random.uniform(-0.1, 0.1)
        self.vx += wind

        # Thrust (fuel-based)
        if self.fuel > 0:
            if action == 0:
                self.angular_velocity += 2
                self.fuel -= 0.5
            elif action == 1:
                self.angular_velocity -= 2
                self.fuel -= 0.5
            elif action == 2:
                self.vx -= 0.5
                self.fuel -= 0.5
            elif action == 3:
                self.vx += 0.5
                self.fuel -= 0.5

        # Position Update
        self.x += self.vx
        self.y += self.vy
        self.angle += self.angular_velocity

        # Crash Conditions
        if abs(self.angle) > 90:
            self.crashed = True
            return -20

        if self.x < 0 or self.x > WIDTH or self.y < 0:
            self.crashed = True
            return -15

        # Landing Check
        if self.y >= HEIGHT - GROUND_HEIGHT:
            self.y = HEIGHT - GROUND_HEIGHT
            self.vx, self.vy, self.angular_velocity = 0, 0, 0

            if abs(self.angle) < 45:
                self.landed = True
                return self.reward_for_landing()
            else:
                self.crashed = True
                return -10

        # Fuel exhausted penalty
        if self.fuel <= 0:
            return -5

        return 0

    # =========================
    # STATE
    # =========================
    def get_state(self):
        return [
            self.x / WIDTH,
            self.y / HEIGHT,
            self.vx,
            self.vy,
            self.angle / 180,
            self.angular_velocity,
            self.fuel / 100,
            int(self.landed or self.crashed)
        ]

    # =========================
    # DRAW
    # =========================
    def draw(self):
        rotated_rocket = pygame.transform.rotate(rocket_img, -self.angle)
        rect = rotated_rocket.get_rect(center=(self.x, self.y))
        screen.blit(rotated_rocket, rect.topleft)

    # =========================
    # LANDING REWARD
    # =========================
    def reward_for_landing(self):

        target_x = WIDTH // 2
        horizontal_distance = abs(self.x - target_x)

        if horizontal_distance <= 10:
            return 100
        elif horizontal_distance <= 30:
            return 50
        elif horizontal_distance <= 60:
            return 20
        else:
            return 5

    # =========================
    # ROTATION PENALTY
    # =========================
    def apply_rotation_penalty(self):
        if abs(self.angle) > 60:
            return -10
        elif abs(self.angle) > 30:
            return -3
        return -0.1

    # =========================
    # STABILIZATION REWARD
    # =========================
    def stabilize_rotation_reward(self):
        if abs(self.angle) < 5:
            return 5
        return 0