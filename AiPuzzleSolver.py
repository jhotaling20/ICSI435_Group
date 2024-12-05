import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU, BatchNormalization
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pygame
from queue import PriorityQueue

# Constants
PUZZLE_SIZE = 3
SEED_SIZE = 100
EPOCHS = 1000
BATCH_SIZE = 32
def generate_puzzle_batch(batch_size):
    puzzles = []
    for _ in range(batch_size):
        puzzle = np.arange(PUZZLE_SIZE * PUZZLE_SIZE)
        np.random.shuffle(puzzle)
        puzzle = puzzle.reshape((PUZZLE_SIZE, PUZZLE_SIZE))
        puzzles.append(puzzle)
    return np.array(puzzles)
# Check if the puzzle is solvable
def is_solvable(puzzle):
    one_d_puzzle = puzzle.flatten()
    inversions = 0
    for i in range(len(one_d_puzzle)):
        for j in range(i + 1, len(one_d_puzzle)):
            if one_d_puzzle[i] > one_d_puzzle[j] and one_d_puzzle[i] != 0 and one_d_puzzle[j] != 0:
                inversions += 1
    return inversions % 2 == 0

# Generate a solvable puzzle
def generate_solvable_puzzle():
    while True:
        puzzle = np.arange(PUZZLE_SIZE * PUZZLE_SIZE)  # Create numbers 0 to 8
        np.random.shuffle(puzzle)  # Shuffle them randomly
        puzzle = puzzle.reshape((PUZZLE_SIZE, PUZZLE_SIZE))  # Reshape into a grid
        if is_solvable(puzzle):  # Check if the generated puzzle is solvable
            return puzzle

# A* Solver
goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])

def heuristic(state, goal=goal_state):
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i, j] != 0:
                goal_pos = np.argwhere(goal == state[i, j])[0]
                distance += abs(i - goal_pos[0]) + abs(j - goal_pos[1])
    return distance

def a_star_search(start):
    start_tuple = tuple(start.flatten())
    goal_tuple = tuple(goal_state.flatten())

    frontier = PriorityQueue()
    frontier.put((0 + heuristic(start), 0, start_tuple))

    came_from = {}
    cost_so_far = {}

    came_from[start_tuple] = None
    cost_so_far[start_tuple] = 0

    while not frontier.empty():
        _, current_cost, current = frontier.get()
        current_state = np.array(current).reshape(3, 3)

        if current == goal_tuple:
            break

        empty_pos = tuple(np.argwhere(current_state == 0)[0])
        x, y = empty_pos

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in moves:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 3 and 0 <= new_y < 3:
                new_state = current_state.copy()
                new_state[x, y], new_state[new_x, new_y] = new_state[new_x, new_y], new_state[x, y]
                new_tuple = tuple(new_state.flatten())

                new_cost = current_cost + 1
                if new_tuple not in cost_so_far or new_cost < cost_so_far[new_tuple]:
                    cost_so_far[new_tuple] = new_cost
                    priority = new_cost + heuristic(new_state)
                    frontier.put((priority, new_cost, new_tuple))
                    came_from[new_tuple] = current

    current = goal_tuple
    path = []
    while current != start_tuple:
        path.append(np.array(current).reshape(3, 3))
        current = came_from[current]
    path.append(start)
    path.reverse()

    return path, cost_so_far[goal_tuple]

# GAN Generator and Discriminator
def build_generator(seed_size):
    model = Sequential()
    model.add(Dense(128, input_dim=seed_size))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(PUZZLE_SIZE * PUZZLE_SIZE, activation='softmax'))
    model.add(Reshape((PUZZLE_SIZE, PUZZLE_SIZE)))
    return model

def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(PUZZLE_SIZE, PUZZLE_SIZE)))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Training GAN
def train_gan(generator, discriminator, dataset, epochs, batch_size, seed_size):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    generator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)

    @tf.function
    def train_step(puzzles):
        noise = tf.random.normal([batch_size, seed_size])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_puzzles = generator(noise, training=True)
            real_output = discriminator(puzzles, training=True)
            fake_output = discriminator(generated_puzzles, training=True)

            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
            disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + \
                        cross_entropy(tf.zeros_like(fake_output), fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss

    for epoch in range(epochs):
        for puzzle_batch in dataset:
            train_step(puzzle_batch)

# Display Puzzle in Pygame
def render_puzzle(fig, puzzle):
    ax = fig.gca()
    ax.clear()
    for i in range(PUZZLE_SIZE):
        for j in range(PUZZLE_SIZE):
            num = int(puzzle[i, j])
            color = 'lightblue' if num != 0 else 'white'
            ax.add_patch(plt.Rectangle((j, PUZZLE_SIZE - i - 1), 1, 1, facecolor=color, edgecolor='black'))
            if num != 0:
                ax.text(j + 0.5, PUZZLE_SIZE - i - 0.5, str(num), ha='center', va='center', fontsize=20)
    ax.set_xlim(0, PUZZLE_SIZE)
    ax.set_ylim(0, PUZZLE_SIZE)
    ax.set_xticks(np.arange(0, PUZZLE_SIZE, 1))
    ax.set_yticks(np.arange(0, PUZZLE_SIZE, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')

def pygame_interface(initial_path):
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Puzzle Solver")
    clock = pygame.time.Clock()

    fig = plt.figure(figsize=(6, 6))
    canvas = FigureCanvas(fig)

    solution_path = initial_path  # Start with the initial path
    step_index = 0

    def generate_new_puzzle():
        """Generate a new solvable puzzle and solve it."""
        new_puzzle = generate_solvable_puzzle()
        new_path, _ = a_star_search(new_puzzle)
        return new_path

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                # Go forward a step
                if event.key == pygame.K_RIGHT:
                    step_index = min(step_index + 1, len(solution_path) - 1)

                # Go backward a step
                elif event.key == pygame.K_LEFT:
                    step_index = max(step_index - 1, 0)

                # Generate a new puzzle
                elif event.key == pygame.K_r:
                    solution_path = generate_new_puzzle()
                    step_index = 0

        # Render the current step
        render_puzzle(fig, solution_path[step_index])
        canvas.draw()
        raw_data = canvas.tostring_rgb()
        size = canvas.get_width_height()

        pygame_surface = pygame.image.fromstring(raw_data, size, "RGB")
        screen.blit(pygame_surface, (0, 0))

        # Display instructions
        font = pygame.font.Font(None, 36)
        instructions = [
            "Press RIGHT to go to the next step",
            "Press LEFT to go to the previous step",
            "Press R to generate a new puzzle"
        ]
        for i, text in enumerate(instructions):
            label = font.render(text, True, (0, 0, 0))
            screen.blit(label, (10, 10 + i * 30))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# Main Execution
puzzle_data = generate_puzzle_batch(1000)
dataset = tf.data.Dataset.from_tensor_slices(puzzle_data).batch(BATCH_SIZE)

generator = build_generator(SEED_SIZE)
discriminator = build_discriminator()

train_gan(generator, discriminator, dataset, EPOCHS, BATCH_SIZE, SEED_SIZE)

noise = tf.random.normal([1, SEED_SIZE])
generated_puzzle = generator(noise, training=False).numpy().reshape((PUZZLE_SIZE, PUZZLE_SIZE))

# Convert the softmax output to a valid permutation
generated_puzzle = np.argsort(generated_puzzle.flatten())  # Ensure valid permutation
generated_puzzle = generated_puzzle.reshape((PUZZLE_SIZE, PUZZLE_SIZE))
if not is_solvable(generated_puzzle):
    generated_puzzle = generate_solvable_puzzle()

solution_path, _ = a_star_search(generated_puzzle)

pygame_interface(solution_path)
