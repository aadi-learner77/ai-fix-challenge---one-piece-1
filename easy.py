import pygame
import json
import random
import sys
from collections import deque
import heapq # ADDED: Import for the priority queue used in A*


# Initialize Pygame
pygame.init()

# Constants
GRID_ROWS = 5
GRID_COLS = 10
CELL_SIZE = 80
WINDOW_WIDTH = GRID_COLS * CELL_SIZE
WINDOW_HEIGHT = GRID_ROWS * CELL_SIZE + 100
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
BROWN = (139, 69, 19)

# Load team configuration
def load_config():
    try:
        # NOTE: This assumes 'team_config.json' exists in the same directory.
        # If it doesn't, this function will exit the program.
        with open('team_config.json', 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print("Error: team_config.json not found! Please create one.")
        # Example minimal config if one is needed for testing:
        # {"team_id": "YourTeam", "seed": 42, "grid_config": {"traffic_lights": 3, "cows": 2, "pits": 3}}
        sys.exit(1)

# World class
class BangaloreWumpusWorld:
    def __init__(self, config):
        self.config = config
        self.seed = config['seed']
        random.seed(self.seed)

        # Initialize grid
        self.grid = [[{'type': 'empty', 'percepts': [], 'weight': random.randint(1,15)}
             for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]


        # Agent starts at bottom-left diagonal (0, GRID_ROWS - 1)
        self.agent_start = (0, GRID_ROWS - 1)
        self.agent_pos = list(self.agent_start)
        self.agent_path = []

        # Game state
        self.game_over = False
        self.game_won = False
        self.message = ""

        # Goal position (will be set in _generate_world)
        self.goal_pos = None

        # Generate world
        self._generate_world()

    def _generate_world(self):
        """Generate random world elements based on config"""
        num_traffic_lights = self.config['grid_config']['traffic_lights']
        num_cows = self.config['grid_config']['cows']
        num_pits = self.config['grid_config']['pits']

        # Generate available positions (exclude agent start)
        available_positions = [(x, y) for x in range(GRID_COLS) for y in range(GRID_ROWS)
                               if (x, y) != tuple(self.agent_start)]

        random.shuffle(available_positions)

        # Place traffic lights
        for i in range(num_traffic_lights):
            if available_positions:
                pos = available_positions.pop()
                self.grid[pos[1]][pos[0]]['type'] = 'traffic_light'

        # Place cows
        for i in range(num_cows):
            if available_positions:
                pos = available_positions.pop()
                self.grid[pos[1]][pos[0]]['type'] = 'cow'

        # Place pits
        for i in range(num_pits):
            if available_positions:
                pos = available_positions.pop()
                self.grid[pos[1]][pos[0]]['type'] = 'pit'

        # Place goal
        if available_positions:
            goal_pos = available_positions.pop()
            self.grid[goal_pos[1]][goal_pos[0]]['type'] = 'goal'
            self.goal_pos = goal_pos
        else:
            # Fallback if no space for goal (unlikely for 5x10 grid)
            print("Error: Could not place goal. Exiting.")
            sys.exit(1)


        # Generate percepts
        self._generate_percepts()

    def _generate_percepts(self):
        """Generate percepts for all cells based on adjacent elements"""
        for y in range(GRID_ROWS):
            for x in range(GRID_COLS):
                neighbors = self._get_neighbors(x, y)

                for nx, ny in neighbors:
                    cell_type = self.grid[ny][nx]['type']

                    if cell_type == 'pit':
                        if 'breeze' not in self.grid[y][x]['percepts']:
                            self.grid[y][x]['percepts'].append('breeze')

                    elif cell_type == 'cow':
                        if 'moo' not in self.grid[y][x]['percepts']:
                            self.grid[y][x]['percepts'].append('moo')

                    elif cell_type == 'traffic_light':
                        if 'light' not in self.grid[y][x]['percepts']:
                            self.grid[y][x]['percepts'].append('light')

    def _get_neighbors(self, x, y):
        """Get valid adjacent neighbors (no diagonals - up, down, left, right only)"""
        neighbors = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_COLS and 0 <= ny < GRID_ROWS:
                neighbors.append((nx, ny))

        return neighbors

    def get_cell_cost(self, x, y):
        """Determine the movement cost for a cell (used in A*)"""
        cell_type = self.grid[y][x]['type']
        if cell_type in ['pit', 'cow']:
            # Impassable obstacle - return infinite cost
            return float('inf')
        elif cell_type == 'traffic_light':
            # Higher cost for traffic light (simulates waiting)
            return 5
        else:
            # Base cost for empty or goal cells
            return 1

    def move_agent(self, new_x, new_y):
        """Move agent to new position and handle interactions"""
        if self.game_over or self.game_won:
            return

        # Check bounds (already checked by A*, but good for manual movement)
        if not (0 <= new_x < GRID_COLS and 0 <= new_y < GRID_ROWS):
            return

        # Only allow orthogonal movement
        dx = abs(new_x - self.agent_pos[0])
        dy = abs(new_y - self.agent_pos[1])
        if dx + dy != 1 and self.agent_pos != self.agent_start: # Allow movement to start for cow reset
            return

        # Check if the move is to a cow or pit (which should have been avoided by A*)
        cell_type = self.grid[new_y][new_x]['type']

        if cell_type == 'cow':
            self.message = "Moo! Cow encountered - returning to start!"
            self.agent_pos = list(self.agent_start)
            self.agent_path = []
            return # Do not record cow position in path before reset

        elif cell_type == 'pit':
            self.message = "Game Over - Fell into a pit!"
            self.game_over = True
            return

        # Move is valid and safe
        self.agent_pos = [new_x, new_y]
        self.agent_path.append((new_x, new_y))

        # Handle other interactions after successful move
        if cell_type == 'traffic_light':
            self.message = "Waiting at traffic signal..."
            # Simulating the delay (visually the agent is in the cell for one frame)
            # In a real-time system, a delay mechanism would be needed.
            # self._simulate_traffic_delay() # Removed as it freezes Pygame
            pass

        elif cell_type == 'goal':
            self.message = "Goal Reached! You won!"
            self.game_won = True
        
        elif cell_type == 'empty':
             self.message = f"Moved to ({new_x}, {new_y})"


    def _simulate_traffic_delay(self):
        """Simulate traffic light delay using nested loop"""
        # NOTE: This function is generally bad practice in a Pygame loop
        # as it will halt the main thread and freeze the display.
        # It's kept here as per the original structure but should ideally be
        # handled using Pygame's timer or frame rate.
        delay = 0
        for i in range(1000):
            for j in range(10000):
                delay += 1

    def get_current_percepts(self):
        """Get percepts at current agent position"""
        x, y = self.agent_pos
        return self.grid[y][x]['percepts']

    # =========================================================================
    # STUDENTS' A* IMPLEMENTATION
    # =========================================================================
    def manhattan_distance(self, p1, p2):
        """Heuristic: Manhattan distance between two points (x1, y1) and (x2, y2)"""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def find_path_astar(self):
        """
        Implementation of A* pathfinding algorithm
        """
        start = tuple(self.agent_pos)
        goal = self.goal_pos

        if start == goal:
            return []

        # Priority Queue: (f_score, g_score, x, y)
        open_set = [(0 + self.manhattan_distance(start, goal), 0, start[0], start[1])]

        # g_score: cost from start to node (x, y)
        g_score = {start: 0}
        
        # came_from: (x, y) -> parent (px, py)
        came_from = {}
        
        while open_set:
            # Pop node with the lowest f_score
            f, g, x, y = heapq.heappop(open_set)
            current = (x, y)

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                self.message = "A* Path Found!"
                # The path should not include the current agent position, 
                # as the first move is *to* the first element of the path.
                return path 

            neighbors = self._get_neighbors(x, y)
            for nx, ny in neighbors:
                neighbor = (nx, ny)

                # Get the movement cost to enter the neighbor cell
                move_cost = self.get_cell_cost(nx, ny)

                # Skip impassable obstacles (Pits and Cows)
                if move_cost == float('inf'):
                    continue

                # Calculate tentative g_score (cost from start to neighbor)
                tentative_g_score = g_score.get(current, float('inf')) + move_cost

                # If this path to neighbor is better than any previously found one
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    # Record the path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    
                    # Calculate f_score (f = g + h)
                    h_score = self.manhattan_distance(neighbor, goal)
                    f_score = tentative_g_score + h_score
                    
                    # Add/update neighbor in the open set
                    heapq.heappush(open_set, (f_score, tentative_g_score, nx, ny))

        # If loop finishes without reaching the goal
        self.message = "Path Not Found"
        return None

    # =========================================================================
    # END OF STUDENTS' A* IMPLEMENTATION
    # =========================================================================

    def execute_path(self, path):
        """Execute a computed path step by step"""
        if path is None:
            return

        # NOTE: Using a delay here to make the execution visible in Pygame
        # A simple iteration will execute too fast to see in the rendering loop.
        for x, y in path:
            self.move_agent(x, y)
            # Short yield/delay for visual update in Pygame:
            pygame.time.delay(200) 
            pygame.event.pump() # Process events (like quit) during delay

            if self.game_over or self.game_won:
                break

# Pygame rendering
class GameRenderer:
    def __init__(self, world):
        self.world = world
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Bangalore Wumpus World - AI CODEFIX 2025")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

    def draw_grid(self):
        """Draw the grid lines"""
        for x in range(0, WINDOW_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (x, 0), (x, WINDOW_HEIGHT - 100), 2)
        for y in range(0, WINDOW_HEIGHT - 100, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (0, y), (WINDOW_WIDTH, y), 2)

    def draw_cell_contents(self):
        """Draw contents of each cell"""
        for y in range(GRID_ROWS):
            for x in range(GRID_COLS):
                cell = self.world.grid[y][x]
                px = x * CELL_SIZE
                py = y * CELL_SIZE

                # Draw cell type
                if cell['type'] == 'traffic_light':
                    pygame.draw.circle(self.screen, RED, (px + CELL_SIZE//2, py + CELL_SIZE//2), 20)
                    text = self.small_font.render("SIGNAL", True, WHITE)
                    self.screen.blit(text, (px + 15, py + 55))

                elif cell['type'] == 'cow':
                    pygame.draw.rect(self.screen, BROWN, (px + 20, py + 20, 40, 40))
                    text = self.small_font.render("COW", True, WHITE)
                    self.screen.blit(text, (px + 25, py + 30))

                elif cell['type'] == 'pit':
                    pygame.draw.circle(self.screen, BLACK, (px + CELL_SIZE//2, py + CELL_SIZE//2), 25)
                    text = self.small_font.render("PIT", True, WHITE)
                    self.screen.blit(text, (px + 28, py + 30))

                elif cell['type'] == 'goal':
                    pygame.draw.rect(self.screen, GREEN, (px + 15, py + 15, 50, 50))
                    text = self.small_font.render("GOAL", True, BLACK)
                    self.screen.blit(text, (px + 20, py + 30))

                # Draw percepts (small indicators)
                percept_y_offset = 10
                if 'breeze' in cell['percepts']:
                    text = self.small_font.render("~", True, BLUE)
                    self.screen.blit(text, (px + 5, py + percept_y_offset))
                    percept_y_offset += 15

                if 'moo' in cell['percepts']:
                    text = self.small_font.render("M", True, BROWN)
                    self.screen.blit(text, (px + 5, py + percept_y_offset))
                    percept_y_offset += 15

                if 'light' in cell['percepts']:
                    text = self.small_font.render("L", True, ORANGE)
                    self.screen.blit(text, (px + 5, py + percept_y_offset))
        
        # Draw the A* calculated path (optional visualization)
        if self.world.game_over is False and self.world.game_won is False:
            self._draw_path(self.world.agent_path)


    def _draw_path(self, path):
        """Draw the path taken by the agent so far"""
        if len(path) > 1:
            points = []
            for x, y in path:
                # Convert grid coordinates to pixel coordinates for the center of the cell
                px = x * CELL_SIZE + CELL_SIZE // 2
                py = y * CELL_SIZE + CELL_SIZE // 2
                points.append((px, py))
            
            # Draw the path as a line
            pygame.draw.lines(self.screen, YELLOW, False, points, 3)
            # Draw circles on each point for clarity
            for px, py in points:
                pygame.draw.circle(self.screen, YELLOW, (px, py), 5)


    def draw_agent(self):
        """Draw the agent"""
        x, y = self.world.agent_pos
        px = x * CELL_SIZE + CELL_SIZE // 2
        py = y * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(self.screen, YELLOW, (px, py), 15)
        pygame.draw.circle(self.screen, BLACK, (px, py), 15, 2)

        # Draw eyes
        pygame.draw.circle(self.screen, BLACK, (px - 5, py - 3), 3)
        pygame.draw.circle(self.screen, BLACK, (px + 5, py - 3), 3)

    def draw_info(self):
        """Draw info panel at bottom"""
        info_y = WINDOW_HEIGHT - 100
        pygame.draw.rect(self.screen, GRAY, (0, info_y, WINDOW_WIDTH, 100))

        # Current position
        pos_text = self.font.render(f"Position: {self.world.agent_pos}", True, BLACK)
        self.screen.blit(pos_text, (10, info_y + 10))

        # Percepts
        percepts = self.world.get_current_percepts()
        percept_text = self.font.render(f"Percepts: {', '.join(percepts) if percepts else 'None'}", True, BLACK)
        self.screen.blit(percept_text, (10, info_y + 35))

        # Message
        msg_text = self.font.render(self.world.message, True, RED if self.world.game_over else GREEN)
        self.screen.blit(msg_text, (10, info_y + 60))

    def render(self):
        """Main render function"""
        self.screen.fill(WHITE)
        self.draw_grid()
        self.draw_cell_contents()
        self.draw_agent()
        self.draw_info()
        pygame.display.flip()
        self.clock.tick(FPS)

# Main game loop
def main():
    config = load_config()
    world = BangaloreWumpusWorld(config)
    renderer = GameRenderer(world)

    print("=== Bangalore Wumpus World ===")
    print(f"Team ID: {config['team_id']}")
    print(f"Agent Start: {world.agent_start}")
    print(f"Goal Position: {world.goal_pos}")
    print("\nControls:")
    print("- Arrow keys: Manual movement")
    print("- SPACE: Execute A* pathfinding")
    print("- R: Reset world")
    print("- ESC: Quit")

    running = True
    while running:
        # Manual movement variables
        current_x, current_y = world.agent_pos
        new_x, new_y = current_x, current_y

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    # Reset world
                    world = BangaloreWumpusWorld(config)
                    renderer.world = world

                elif event.key == pygame.K_SPACE:
                    # Execute A* pathfinding
                    if not world.game_over and not world.game_won:
                        print("\n=== Executing A* Pathfinding ===")
                        # Clear path history to show only A* calculated path
                        world.agent_path = [tuple(world.agent_pos)]
                        
                        path = world.find_path_astar()
                        if path:
                            print(f"Path found (length {len(path)}): {path[0]} -> ... -> {path[-1]}")
                            # Execute path, step by step
                            world.execute_path(path)
                        else:
                            print("Path not found or A* failed.")
                
                # Manual movement handling
                elif not world.game_over and not world.game_won:
                    if event.key == pygame.K_UP:
                        new_y -= 1
                    elif event.key == pygame.K_DOWN:
                        new_y += 1
                    elif event.key == pygame.K_LEFT:
                        new_x -= 1
                    elif event.key == pygame.K_RIGHT:
                        new_x += 1
                
                # Execute manual move
                if (new_x, new_y) != (current_x, current_y):
                    world.move_agent(new_x, new_y)


        renderer.render()

    pygame.quit()

if __name__ == "__main__":
    main()
