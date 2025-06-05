# nuruomino.py: Optimized implementation for AI 2024/2025 project
# Grupo 18:
# 109370 Miguel Trêpa
# 90173 Rafael Ferreira

from sys import stdin
import numpy as np
import time
from search import Problem, Node, depth_first_tree_search, greedy_search, astar_search, breadth_first_tree_search, \
    depth_first_graph_search, iterative_deepening_search

# Precomputed pieces with tuples for faster hashing and immutability
PIECES = {
    'L': [
        ((0,0), (0,1), (1,0), (2,0)),
        ((0,0), (0,-1), (1,0), (2,0)),
        ((0,0), (1,0), (0,1), (0,2)),
        ((0,0), (1,0), (0,-1), (0,-2)),
        ((0,0), (-1,0), (0,1), (0,2)),
        ((0,0), (-1,0), (0,-1), (0,-2)),
        ((0,0), (-1,0), (-2,0), (0,1)),
        ((0,0), (-1,0), (-2,0), (0,-1)),
    ],
    'I': [
        ((0,0), (1,0), (2,0), (3,0)),
        ((0,0), (0,1), (0,2), (0,3))
    ],
    'T': [
        ((0,0), (0,-1), (0,1), (1,0)),
        ((0,0), (-1,0), (1,0), (0,1)),
        ((0,0), (0,-1), (0,1), (-1,0)),
        ((0,0), (-1,0), (1,0), (0,-1))
    ],
    'S': [
        ((0,0), (0,1), (1,-1), (1,0)),
        ((0,0), (0,-1), (1,0), (1,1)),
        ((0,0), (1,0), (1,1), (2,1)),
        ((0,0), (1,0), (1,-1), (2,-1))
    ]
}

# Precompute adjacent directions as tuples for immutability
ADJACENT_4 = ((-1,0), (1,0), (0,-1), (0,1))
# ADJACENT_8 is not currently used in the optimized code, but kept for reference
ADJACENT_8 = ((-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1))

# Precompute the set of marks for faster lookups
MARKS_SET = {'L', 'I', 'T', 'S'}

class NuruominoState:
    state_id = 0
    
    def __init__(self, board: 'Board'):
        self.board = board
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1
        # Cache hash for faster comparisons in graph search
        self._hash = None
    
    def __lt__(self, other: 'NuruominoState') -> bool:
        """This method is used in case of ties in the open list of informed searches."""
        return self.id < other.id
    
    def __hash__(self):
        """Generates a hash for the state, crucial for graph search to detect visited states."""
        if self._hash is None:
            # Combine hash of the board array (converted to bytes) and the haspiece list (as tuple)
            self._hash = hash((self.board.board.tobytes(), tuple(self.board.haspiece)))
        return self._hash
    
    def __eq__(self, other):
        """Checks if two states are equal, also crucial for graph search."""
        if not isinstance(other, NuruominoState):
            return False
        # Compare board arrays and haspiece lists
        return (np.array_equal(self.board.board, other.board.board) and 
                self.board.haspiece == other.board.haspiece)

class Board:
    """
    Internal representation of a Nuruomino puzzle board.

    Args:
        board (np.ndarray): The matrix with region identifiers (or placed pieces).
        haspiece (list[bool]): Flags indicating if each region already has a piece.
        regions (list[int]): List of present regions (numerical IDs).
        cells (tuple[tuple[tuple]]): Tuple of tuples of (row, col) positions for each region.
    """
    def __init__(self, board=None, haspiece=None, regions=None, cells=None):
        if board is None:
            # Default to a 1x1 board with region 1, ensuring dtype=object
            board = np.array([[1]], dtype=object)
        
        # ALWAYS ensure self.board is an object array to hold both ints and strings
        self.board = board if board.dtype == object else board.astype(object)

        self.height, self.width = self.board.shape
        
        if haspiece is None:
            # Identify original numerical regions from the board
            # Filter out any piece marks that might have already been on the board (e.g., from a partial solution).
            # Ensure unique_regions are sorted numerically if they are numbers.
            unique_board_elements = np.unique(self.board)
            original_unique_regions_numeric = sorted([
                int(x) for x in unique_board_elements if isinstance(x, (int, np.integer)) or (isinstance(x, str) and x.isdigit())
            ])
            self.regions = original_unique_regions_numeric # Store as a list of integers
            
            region_count = len(self.regions)
            self.haspiece = [False] * region_count
            
            self.cells = [[] for _ in range(region_count)]
            # Create a mapping from region ID to its 0-based internal index for faster lookups
            self.region_id_to_idx = {region_id: i for i, region_id in enumerate(self.regions)}
            
            # Precompute cells for each region
            for r in range(self.height):
                for c in range(self.width):
                    cell_val = self.board[r, c]
                    # Convert string cell_val to int if it's a digit for lookup
                    if isinstance(cell_val, str) and cell_val.isdigit():
                        cell_val = int(cell_val)
                    
                    # Only map original region IDs, not placed pieces
                    if isinstance(cell_val, (int, np.integer)) and cell_val in self.region_id_to_idx:
                        self.cells[self.region_id_to_idx[cell_val]].append((r, c))
            
            self.cells = tuple(tuple(l) for l in self.cells) # Convert to tuple of tuples for immutability
        else:
            # If haspiece, regions, and cells are provided (e.g., from a place_fast call)
            self.haspiece = list(haspiece)
            self.cells = cells
            self.regions = regions
            self.region_id_to_idx = {region_id: i for i, region_id in enumerate(self.regions)}
        
    def __str__(self):
        """Returns a textual representation of the board."""
        return '\n'.join('\t'.join(str(cell) for cell in row) for row in self.board)
    
    def makes_2x2_fast(self, piece_positions: list[tuple[int, int]]) -> bool:
        """
        Checks if placing the piece would form a 2x2 square.
        Uses optimized set operations and direct board lookups.
        """
        piece_set = set(piece_positions)
        
        # Iterate over each cell of the piece and its immediate neighbors
        # to find potential top-left corners of 2x2 blocks.
        checked_2x2_corners = set()
        for r, c in piece_positions:
            for dr_corner, dc_corner in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
                block_r, block_c = r + dr_corner, c + dc_corner
                
                # Check if this top-left corner is within valid bounds for a 2x2 block
                if not (0 <= block_r < self.height - 1 and 0 <= block_c < self.width - 1):
                    continue
                
                # Avoid redundant checks for the same 2x2 block
                if (block_r, block_c) in checked_2x2_corners:
                    continue
                checked_2x2_corners.add((block_r, block_c))

                # Define the four cells of the potential 2x2 block
                block_cells = {
                    (block_r, block_c), (block_r + 1, block_c),
                    (block_r, block_c + 1), (block_r + 1, block_c + 1)
                }
                
                count_filled = 0
                for cell_r, cell_c in block_cells:
                    if (cell_r, cell_c) in piece_set:
                        count_filled += 1
                    else:
                        # Check if the existing cell is an already placed piece mark
                        if self.board[cell_r, cell_c] in MARKS_SET:
                            count_filled += 1
                
                if count_filled == 4:
                    return True # A 2x2 square would be formed
        return False
    
    def can_place_fast(self, shape: tuple[tuple[int, int]], origin: tuple[int, int], region_id: int, mark: str) -> bool:
        """
        Verifies if a piece can be placed at 'origin' within 'region_id' with 'mark',
        respecting all Nuruomino rules. Optimized for speed.
        """
        
        # Check if region is already filled
        region_idx = self.region_id_to_idx.get(region_id)
        if region_idx is None or self.haspiece[region_idx]:
            return False
        
        target_region_int = int(region_id)

        piece_positions = []
        has_frontier = False
        
        for dr, dc in shape:
            r, c = origin[0] + dr, origin[1] + dc
            
            # Rule 1: Piece must be within board boundaries and within its region
            if not (0 <= r < self.height and 0 <= c < self.width):
                return False # Out of bounds
            
            current_cell_val = self.board[r, c]
            # Convert string cell_val to int if it's a digit for comparison
            if isinstance(current_cell_val, str) and current_cell_val.isdigit():
                current_cell_val = int(current_cell_val)

            if not (isinstance(current_cell_val, (int, np.integer)) and current_cell_val == target_region_int):
                return False # Outside the designated region or already occupied by another piece

            piece_positions.append((r, c))
            
            # Rule 2: No two pieces of the same type can be adjacent (4-directionally)
            for adj_dr, adj_dc in ADJACENT_4:
                adj_r_abs, adj_c_abs = r + adj_dr, c + adj_dc
                if (0 <= adj_r_abs < self.height and 0 <= adj_c_abs < self.width and 
                    self.board[adj_r_abs, adj_c_abs] == mark):
                    return False # Adjacent to an identical piece
            
            # Rule 4: Frontier check - A piece must be adjacent to an already placed piece,
            # unless it's the first piece on the board.
            # Only check if 'has_frontier' isn't already true to save time.
            if not has_frontier:
                for adj_dr, adj_dc in ADJACENT_4:
                    adj_r_abs, adj_c_abs = r + adj_dr, c + adj_dc
                    if (0 <= adj_r_abs < self.height and 0 <= adj_c_abs < self.width and 
                        self.board[adj_r_abs, adj_c_abs] in MARKS_SET): # Check if an adjacent cell has a placed piece
                        has_frontier = True
                        break # Found an adjacent piece for *this* cell of the proposed piece.
                                # No need to check other neighbors for this cell.
        
        # Rule 3: Check for 2x2 square formation
        if self.makes_2x2_fast(piece_positions):
            return False
        
        # Final Frontier rule application:
        # If no pieces are on the board yet, any placement is valid.
        # Otherwise, 'has_frontier' must be true (meaning the piece touches an existing one).
        return has_frontier or not any(self.haspiece)
    
    def place_fast(self, shape: tuple[tuple[int, int]], origin: tuple[int, int], mark: str) -> 'Board':
        """
        Returns a new Board instance with the piece placed, marked with 'mark'.
        Optimized to avoid recomputing precomputed attributes.
        """
        new_board_array = self.board.copy()
        for dr, dc in shape:
            new_board_array[origin[0] + dr, origin[1] + dc] = mark
        
        new_haspiece = self.haspiece.copy()
        
        # Get the original region ID from the board *before* placing the piece
        # This ensures we correctly identify which region is being filled.
        region_id_at_origin = self.board[origin[0], origin[1]]
        # Convert to int if it's a string digit
        if isinstance(region_id_at_origin, str) and region_id_at_origin.isdigit():
            region_id_at_origin = int(region_id_at_origin)

        region_idx = self.region_id_to_idx[region_id_at_origin]
        new_haspiece[region_idx] = True # Mark the corresponding region as filled

        # Pass all precomputed data to the new Board instance
        return Board(new_board_array, new_haspiece, self.regions, self.cells)
    
    @staticmethod
    def parse_instance() -> 'Board':
        """
        Reads the puzzle instance from standard input and returns a Board instance.
        """
        lines = []
        for line in stdin:
            line = line.strip()
            if line:
                lines.append(line.split())
        
        if not lines:
            return Board() # Return a default empty board if no input
        
        # When creating the numpy array from parsed lines, explicitly set dtype=object
        board_data = np.array(lines, dtype=object)
        
        return Board(board_data)

class Nuruomino(Problem):
    def __init__(self, board: Board):
        super().__init__(NuruominoState(board))
        self.cnt = 0 # Counter for nodes expanded during search
        
        # Precompute action priorities for sorting
        self.action_priority = {'S': 0, 'T': 1, 'L': 2, 'I': 3}
        
        # Generate all possible (region, shape, origin, mark) tuples once
        # These are potential actions that will be validated by can_place_fast later.
        self._all_potential_actions = self._generate_all_potential_actions(board)

    def _generate_all_potential_actions(self, initial_board: 'Board') -> dict:
        """
        Generates all possible piece placements for each region, sorted by piece priority.
        These are "potential" actions before validation against current board state.
        """
        potential_actions_by_region = {}
        for region_id in initial_board.regions:
            region_idx = initial_board.region_id_to_idx[region_id]
            region_cells = initial_board.cells[region_idx]
            
            actions_for_region = []
            for origin in region_cells:
                for mark, shape_group in PIECES.items():
                    for shape in shape_group:
                        actions_for_region.append((region_id, shape, origin, mark))
            
            # Sort by piece priority within each region's potential actions
            actions_for_region.sort(key=lambda x: self.action_priority[x[3]])
            potential_actions_by_region[region_id] = tuple(actions_for_region) # Store as tuple for immutability
            
        return potential_actions_by_region

    def actions(self, state: NuruominoState) -> list:
        """
        Generates all valid actions (piece placements) from the current state.
        Prioritizes forced moves and regions with fewer placement options.
        """
        board = state.board
        
        # Get IDs of regions that are currently unfilled
        empty_regions_ids = [
            r_id for r_id in board.regions if not board.haspiece[board.region_id_to_idx[r_id]]
        ]
        
        # Optimization: Check for forced moves (regions of exactly 4 cells) first
        for region_id in empty_regions_ids:
            region_idx = board.region_id_to_idx[region_id]
            if len(board.cells[region_idx]) == 4:
                # Iterate through precomputed potential actions for this region
                for action in self._all_potential_actions[region_id]:
                    # Validate the potential action with the current board state
                    if board.can_place_fast(action[1], action[2], action[0], action[3]):
                        return [action] # Forced move found, return it immediately as the only action
        
        # If no forced moves, collect all other valid actions
        valid_actions_by_region_size = [] # Will store (region_size, list_of_valid_actions)
        for region_id in empty_regions_ids:
            region_idx = board.region_id_to_idx[region_id]
            region_size = len(board.cells[region_idx])
            
            current_region_valid_actions = []
            for action in self._all_potential_actions[region_id]:
                # Validate the potential action with the current board state
                if board.can_place_fast(action[1], action[2], action[0], action[3]):
                    current_region_valid_actions.append(action)
            
            if current_region_valid_actions:
                valid_actions_by_region_size.append((region_size, current_region_valid_actions))
        
        # Sort regions by the number of valid actions (most constrained variable heuristic).
        # Smallest region size first, as these tend to have fewer options.
        valid_actions_by_region_size.sort(key=lambda x: x[0])
        
        final_actions_list = []
        for _, actions_list in valid_actions_by_region_size:
            final_actions_list.extend(actions_list) # Flatten the list
            
        return final_actions_list
    
    def result(self, state: NuruominoState, action) -> NuruominoState:
        """Returns the state resulting from executing 'action' on 'state'."""
        # The place_fast method now handles updating the haspiece flags
        # and creating a new Board instance with correctly copied attributes.
        new_board = state.board.place_fast(action[1], action[2], action[3])
        return NuruominoState(new_board)
    
    def goal_test(self, state: NuruominoState) -> bool:
        """Returns True if the given state is a goal state."""
        self.cnt += 1 # Increment node counter
        return all(state.board.haspiece) # True if all regions are filled
    
    def h(self, node: Node) -> float:
        """
        Heuristic function: Estimates the cost to reach the goal from the current state.
        Returns the number of unfilled regions, plus a penalty for small, unfilled regions.
        """
        board = node.state.board
        unfilled_regions_count = sum(1 for has_piece in board.haspiece if not has_piece)
        
        # Add a penalty for regions that are small and thus potentially harder/more restrictive to fill.
        # This can guide the search towards filling larger regions first, which might open up more options
        # or reveal dead ends earlier.
        penalty_for_small_unfilled = 0
        for i, region_id in enumerate(board.regions):
            if not board.haspiece[i]:
                region_size = len(board.cells[i])
                if region_size == 4: # Smallest possible region for a piece
                    penalty_for_small_unfilled += 1.0 # Higher penalty for regions that are exactly 4 cells
                elif region_size <= 8: # Arbitrary threshold for 'small' regions
                    penalty_for_small_unfilled += 0.5
        
        return unfilled_regions_count + (penalty_for_small_unfilled * 0.1) # Small multiplier for the penalty

if __name__ == "__main__":
    # Parse the puzzle instance from standard input
    problem_board = Board.parse_instance()
    
    # Create the problem instance
    problem = Nuruomino(problem_board)
    
    print("Starting Nuruomino search...")
    start_time = time.time()
    
    # Solve using depth-first graph search. This search algorithm is effective
    # for finding *a* solution (not necessarily the shortest path, but here path length
    # corresponds to number of pieces, which is fixed by the puzzle size).
    # The state caching provided by __hash__ and __eq__ in NuruominoState
    # makes graph search much more efficient by avoiding redundant exploration of states.
    result_node = depth_first_graph_search(problem) 
    
    end_time = time.time()
    
    if result_node is not None:
        print("\nSolution Found:")
        print(result_node.state.board)
        print(f"\nNodes expanded: {problem.cnt}")
    else:
        print("\nNo solution found.")
    
    print(f"Time taken: {end_time - start_time:.4f} seconds")