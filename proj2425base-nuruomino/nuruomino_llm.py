import numpy as np
from sys import stdin
from search import Problem, Node, depth_first_tree_search, greedy_search, astar_search, breadth_first_tree_search, \
    depth_first_graph_search, iterative_deepening_search

# Conjunto de peças do jogo Nuruomino
MARKS = {'L', 'I', 'T', 'S'}

# Cada shape é uma lista de offsets (row, col) a partir de uma origem (0, 0)
PIECES = {
    'L': [
        [(0,0), (0,1), (1,0), (2,0)],
        [(0,0), (0,-1), (1,0), (2,0)],
        [(0,0), (1,0), (0,1), (0,2)],
        [(0,0), (1,0), (0,-1), (0,-2)],
        [(0,0), (-1,0), (0,1), (0,2)],
        [(0,0), (-1,0), (0,-1), (0,-2)],
        [(0,0), (-1,0), (-2,0), (0,1)],
        [(0,0), (-1,0), (-2,0), (0,-1)],
    ],
    'I': [
        [(0,0), (1,0), (2,0), (3,0)],
        [(0,0), (0,1), (0,2), (0,3)]
    ],
    'T': [
        [(0,0), (0,-1), (0,1), (1,0)],
        [(0,0), (-1,0), (1,0), (0,1)],
        [(0,0), (0,-1), (0,1), (-1,0)],
        [(0,0), (-1,0), (1,0), (0,-1)]
    ],
    'S': [
        [(0,0), (0,1), (1,-1), (1,0)],
        [(0,0), (0,-1), (1,0), (1,1)],
        [(0,0), (1,0), (1,1), (2,1)],
        [(0,0), (1,0), (1,-1), (2,-1)]
    ]
}

# Direções adjacentes sem diagonal
DIRECTIONS = ((-1,0), (1,0), (0,-1), (0,1))
# Direções adjacentes com diagonal (not used in Nuruomino but kept for completeness)
DIAGONAL_DIRECTIONS = ((-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1))

class NuruominoState:
    state_id = 0

    def __init__(self, board: 'Board'):
        self.board = board
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1
        self._hash = None

    def __lt__(self, other: 'NuruominoState') -> bool:
        return self.id < other.id

    def __hash__(self):
        if self._hash is None:
            board_content_tuple = tuple(tuple(cell for cell in row) for row in self.board.board)
            self._hash = hash((board_content_tuple, tuple(self.board.haspiece)))
        return self._hash

    def __eq__(self, other: 'NuruominoState') -> bool:
        if not isinstance(other, NuruominoState):
            return False
        return (np.array_equal(self.board.board, other.board.board) and
                self.board.haspiece == other.board.haspiece)

class Board:
    def __init__(self, board=None, haspiece=None, regions=None, cells=None):
        if board is None:
            board = np.array([[1]], dtype=object)

        self.board = board
        self.height, self.width = self.board.shape

        if haspiece is None:
            unique_regions = np.unique(board)
            unique_regions_int = sorted([int(ur) for ur in unique_regions])
            self.regions = unique_regions_int
            region_count = len(self.regions)
            self.haspiece = [False] * region_count
            self.cells = [[] for _ in range(region_count)]
            for r in range(self.height):
                for c in range(self.width):
                    self.cells[int(self.board[r, c]) - 1].append((r, c))
            self.cells = tuple(tuple(l) for l in self.cells)
        else:
            self.haspiece = list(haspiece)
            self.cells = cells
            self.regions = regions

    def __str__(self):
        return '\n'.join('\t'.join(str(cell) for cell in row) for row in self.board)

    def makes_2x2(self, piece_coordinates: set[tuple[int, int]]) -> bool:
        """
        Optimized check to see if placing a piece would form a 2x2 square.
        Uses a more efficient approach by checking each potential 2x2 square only once.
        """
        height, width = self.board.shape
        
        # Get all potential top-left corners of 2x2 squares that could involve our piece
        potential_corners = set()
        for r, c in piece_coordinates:
            # For each piece cell, it could be part of 2x2 squares with these top-left corners:
            for dr, dc in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
                tl_r, tl_c = r + dr, c + dc
                if 0 <= tl_r < height - 1 and 0 <= tl_c < width - 1:
                    potential_corners.add((tl_r, tl_c))
        
        # Check each unique corner only once
        for tl_r, tl_c in potential_corners:
            square_cells = [
                (tl_r, tl_c), (tl_r, tl_c + 1),
                (tl_r + 1, tl_c), (tl_r + 1, tl_c + 1)
            ]
            
            # Count filled cells in this 2x2 square
            filled_count = 0
            for sq_r, sq_c in square_cells:
                if ((sq_r, sq_c) in piece_coordinates or 
                    (isinstance(self.board[sq_r, sq_c], str) and self.board[sq_r, sq_c] in MARKS)):
                    filled_count += 1
                else:
                    break  # Early exit if we find an empty cell
            
            if filled_count == 4:
                return True
        
        return False

    def can_place(self, shape: list[tuple[int, int]], origin: tuple[int, int], region_id: int, mark: str) -> bool:
        """
        Highly optimized placement validation with early exits and minimal redundant checks.
        """
        # 1. Cheapest check first: region already filled
        region_index = int(region_id) - 1
        if region_index < 0 or region_index >= len(self.haspiece) or self.haspiece[region_index]:
            return False
        
        height, width = self.board.shape
        current_region_cells_set = set(self.cells[region_index])
        is_first_piece = not any(self.haspiece)
        
        piece_positions = []
        fronter_found = False
        
        # Pre-allocate for adjacency checks (avoid repeated tuple creation)
        directions = DIRECTIONS  # Cache reference
        
        for dr, dc in shape:
            r, c = origin[0] + dr, origin[1] + dc
            
            # 2. Bounds and region membership check
            if not (0 <= r < height and 0 <= c < width and (r, c) in current_region_cells_set):
                return False
            
            # 3. Combined adjacency and fronter check (single loop)
            if not fronter_found or not is_first_piece:  # Skip if both conditions already met
                for adj_dr, adj_dc in directions:
                    adj_r, adj_c = r + adj_dr, c + adj_dc
                    if 0 <= adj_r < height and 0 <= adj_c < width:
                        adj_cell_value = self.board[adj_r, adj_c]
                        
                        # Same-piece adjacency check (most critical)
                        if adj_cell_value == mark:
                            return False
                        
                        # Fronter condition (only check if needed)
                        if not is_first_piece and not fronter_found and adj_cell_value in MARKS:
                            fronter_found = True
            
            piece_positions.append((r, c))
        
        # 4. Fronter validation (only for non-first pieces)
        if not is_first_piece and not fronter_found:
            return False
        
        # 5. 2x2 check last (most expensive)
        if self.makes_2x2(set(piece_positions)):
            return False
        
        return True

    def place(self, shape: list[tuple[int, int]], origin: tuple[int, int], mark) -> 'Board':
        """Retorna uma nova instância de Board com a peça colocada, marcada com `mark`."""
        new_board_array = self.board.copy()
        for dr, dc in shape:
            new_board_array[origin[0] + dr, origin[1] + dc] = mark

        new_haspiece = self.haspiece.copy()

        region_id = self.board[origin[0], origin[1]]
        new_haspiece[int(region_id) - 1] = True

        return Board(new_board_array, new_haspiece, self.regions, self.cells)

    def adjacent_regions(self, region:int, diag: bool) -> list:
        """Devolve uma lista das regiões que fazem fronteira com a região enviada no argumento."""
        regions = set()
        region_cells = self.cells[int(region) - 1]

        # Optimize by checking adjacency for all cells in the region
        for r_cell, c_cell in region_cells:
            for r, c in self.adjacent_positions(r_cell, c_cell, diag):
                if self.board[r, c] != region:
                    regions.add(self.board[r, c])

        return list(regions)

    def adjacent_positions(self, r: int, c: int, diag = False) -> list:
        """
        Retorna uma lista de posições adjacentes da posição (r, c) do tabuleiro.
        Se diag = True, consideramos as diagonais
        """
        adjacent = set()

        # Corrected directions for diag=True and diag=False
        if diag:
            directions = DIAGONAL_DIRECTIONS
        else:
            directions = DIRECTIONS

        if 0 <= r < self.height and 0 <= c < self.width: # Check if r, c are valid starting points
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    adjacent.add((nr, nc))

        return list(adjacent)

    def adjacent_values(self, row:int, col:int, diag: bool) -> list:
        """Devolve os valores das celulas adjacentes à posição, em todas as direções, incluindo diagonais.
        Formato: [LeftTopDiag, , Top, RightTopDiag, Left, Right, , BottomLeftDiag, Bottom, BottomRightDiag]"""
        values = []

        if diag:
            directions = DIAGONAL_DIRECTIONS
        else:
            directions = DIRECTIONS

        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.board.shape[0] and 0 <= c < self.board.shape[1]:
                values.append(self.board[r][c])

        return values

    @staticmethod
    def parse_instance() -> 'Board':
        """
        Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.
        """
        lines = []
        for line in stdin:
            line = line.strip()
            if line:
                lines.append(line.split())

        board_data = np.array(lines, dtype=object)

        return Board(board_data)

class Nuruomino(Problem):
    def __init__(self, board: Board):
        super().__init__(NuruominoState(board))
        # Contador de estados visitados (para debugging)
        self.cnt = 0

        # Atribuir prioridade às peças diferentes
        # (de acordo com o número quadrados que cada peça de facto ocupa,
        # ou seja, tendo em conta o facto de o S e o T, fazerem com que
        # dois quadrados seja impossíveis, o L, um quadrado e o I, 0)
        self.action_priority = {'S': 0, 'T': 1, 'L': 2, 'I': 3}

        # Gerar as ações iniciais (pré-computação)
        self.potential_actions = self._generate_potential_actions(board)

    def _generate_potential_actions(self, board: Board):
        """
        Gera todas as ações que podem ser feitas no tabuleiro, ordenadas
        pela prioridade das peças
        """
        potential_actions = {}
        for region_id in board.regions:
            region = int(region_id) - 1
            region_cells = board.cells[region]

            region_actions = []
            for origin in region_cells:
                for mark, shapes in PIECES.items():
                    for shape in shapes:
                        region_actions.append((region_id, shape, origin, mark))

            # Sort actions for this region based on piece priority.
            # This ensures that for a given region, we try 'S' then 'T' then 'L' then 'I'.
            region_actions.sort(key=lambda x: self.action_priority[x[3]])
            potential_actions[region_id] = region_actions

        return potential_actions

    def actions(self, state: NuruominoState):
        """Gera todas as formas válidas de colocar uma peça no estado atual."""
        board = state.board

        empty_regions_ids = [
            region_id for region_id in board.regions if not board.haspiece[region_id - 1]
        ]

        # Optimization: Try to fill 4-cell regions first as they are forced moves
        for region_id in empty_regions_ids:
            # We already know the size of regions from board.cells in __init__
            if len(board.cells[int(region_id) - 1]) == 4:
                # Iterate through potential actions for this 4-cell region,
                # which are already sorted by piece priority ('S' first).
                for action in self.potential_actions[region_id]:
                    if board.can_place(action[1], action[2], action[0], action[3]):
                        # If a valid action is found for a 4-cell region, it's a strong candidate.
                        # We assume these are highly constrained and returning only one might be a good greedy choice.
                        return [action]

        # If no forced 4-cell region moves, proceed with general action generation
        actions_by_region_size = []
        for region_id in empty_regions_ids:
            region_size = len(board.cells[region_id - 1])
            region_actions = []
            for action in self.potential_actions[region_id]:
                if board.can_place(action[1], action[2], action[0], action[3]):
                    region_actions.append(action)
            if region_actions:
                actions_by_region_size.append((region_size, region_actions))

        actions_by_region_size.sort(key=lambda x: x[0]) # Sort by region size (smallest first)

        actions = []
        for _, actions_list in actions_by_region_size:
            actions.extend(actions_list) # Flatten the list

        return actions

    def result(self, state: NuruominoState, action) -> NuruominoState:
        """Retorna o estado resultante de executar a 'action' sobre 'state'."""
        new_board = state.board.place(action[1], action[2], action[3])
        return NuruominoState(new_board)


    def goal_test(self, state: NuruominoState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        self.cnt += 1
        return all(state.board.haspiece) # Verifica se tem uma peça em todas as regiões

    def h(self, node: Node):
        """Retorna o nº de regiões por preencher no board"""
        regioes_a_preencher = sum(not haspiece for haspiece in node.state.board.haspiece)
        return regioes_a_preencher


if __name__ == "__main__":
    # Ler o tabuleiro do standard input e cria uma instância da classe Board
    problem_board = Board.parse_instance()

    # Criar uma instância do problema
    problem = Nuruomino(problem_board)

    # Criar o estado inicial do problema
    initial_state = NuruominoState(problem_board)

    # Faz a procura
    result = depth_first_graph_search(problem)

    # Retorna o resultado
    if result is not None:
        result = result.state
        print(result.board)
    else:
        print("No solution found.")