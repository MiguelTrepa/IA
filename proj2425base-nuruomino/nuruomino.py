# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 18:
# 109370 Miguel Trêpa
#  90173 Rafael Ferreira


from collections import defaultdict
from sys import stdin
import numpy as np
from search import Problem, Node, depth_first_tree_search, greedy_search, astar_search, breadth_first_tree_search, \
    depth_first_graph_search, iterative_deepening_search

# Each shape is a list of (row, col) offsets from the origin (0, 0)
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

class NuruominoState:
    state_id = 0

    def __init__(self, board: 'Board'):
        self.board = board
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1

    def __lt__(self, other: 'NuruominoState') -> bool:
        """ Este método é utilizado em caso de empate na gestão da lista
        de abertos nas procuras informadas. """
        return self.id < other.id

class Board:
    """
    Representação interna de um tabuleiro do Puzzle Nuruomino.

    Args:
        board (np.ndarray): A matriz com identificadores de regiões.
        haspiece (list[bool]): Flags indicando se cada região já tem uma peça.
        regions (list[str]): Lista das regiões presentes.
    """
    def __init__(self, board=np.array([[1]], dtype=np.int8), haspiece=np.array([1], dtype=np.bool), 
            regions=np.array([1], dtype=np.int8), cells=np.array([[1]])):
        self.board = board
        self.height, self.width = self.board.shape

        if haspiece is None or len(haspiece) == 0:
            unique_regions = np.unique(board)
            try:
                region_count = max(int(x) for x in unique_regions)
            except:
                region_count = len(unique_regions)
            cellsaux = []
            self.haspiece = [np.bool(False)] * region_count
            for i in range(region_count):
                cellsaux.append(self.region_cells(np.str_(i + 1)))
            self.cells = np.array(cellsaux, dtype=object)
        else:
            self.haspiece = list(haspiece)
            self.cells = cells

        if regions is None or len(regions) == 0:
            self.regions = np.unique(board)
        else:
            self.regions = list(regions)

    def __str__(self):
        """Devolve uma representação textual do tabuleiro."""
        formatted_board = ""
        for row in self.board:
            formatted_board += "\t".join(str(cell) for cell in row) + "\n"
        return formatted_board
    
    
    def piece_adjacent_positions(self, origin: tuple[int, int], shape: list[tuple[int, int]], diag = False) -> list:
        """
        Retorna uma lista de posições adjacentes à peça colocada na posição origin.
        A peça é representada por uma lista de tuplas (r, c) que indicam as posições ocupadas.
        """
        adjacent = set()
        
        placed = [(origin[0] + dx, origin[1] + dy) for dx, dy in shape]

        for r, c in placed:
            adjacent_positions = self.adjacent_positions(r, c, diag)
            for pos in adjacent_positions:
                if pos not in placed:
                    adjacent.add(pos)

        return list(adjacent)
    
    def makes_2x2(self, origin: tuple[int, int], shape: list[tuple[int, int]]) -> bool:
        """
        Verifica se a colocação da peça na posição origin forma um quadrado 2x2,
        considerando tanto a peça quanto as células adjacentes a ela.
        Retorna True se formar um quadrado 2x2, False caso contrário.
        """
        temp_board = self.board.copy()
        # Marca as posições da peça com piece_label
        for dr, dc in shape:
            r, c = origin[0] + dr, origin[1] + dc
            if 0 <= r < self.height and 0 <= c < self.width:
                temp_board[r, c] = 'L'
            else:
                return False

        # Define área relevante para verificar 2x2
        piece = set([(origin[0] + dr, origin[1] + dc) for dr, dc in shape])
        surrounds = set(self.piece_adjacent_positions(origin, shape, diag=True))
        greater_area = piece | surrounds

        marks = {'L', 'I', 'T', 'S'}
        for r, c in greater_area:
            # Verifica se é possível formar um quadrado 2x2 a partir de (r, c)
            if r + 1 < self.height and c + 1 < self.width:
                O_SHAPE = {temp_board[r + dr, c + dc] for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]}
                if all(mark in marks for mark in O_SHAPE):
                    return True
        return False
    
    def can_place(self, shape: list[tuple[int, int]], origin: tuple[int, int], region_id: int, mark: str) -> bool:
        """Verifica se é possível colocar a forma no tabuleiro a partir de `origin`, respeitando os limites da região."""
        frontier = False
        marks = np.array(['L', 'I', 'T', 'S'])
        if (self.haspiece[int(region_id) - 1]):
            return False
        for dr, dc in shape:
            r, c = origin[0] + dr, origin[1] + dc
            if not (0 <= r < self.board.shape[0] and 0 <= c < self.board.shape[1]):
                return False  # fora do tabuleiro
            if self.board[r, c] != region_id:
                return False  # fora da região
            if (r - 1 >= 0 and self.board[r - 1, c] == mark):
                return False # peça à esquerda igual
            if (c - 1 >= 0 and self.board[r, c - 1] == mark):
                return False # peça em cima igual
            if (r + 1 < self.board.shape[0] and self.board[r + 1, c] == mark):
                return False # peça à direita igual
            if (c + 1 < self.board.shape[1] and self.board[r, c + 1] == mark):
                return False # peça em baixo igual
            adj_values = self.adjacent_values(r, c, False)
            if any(val in marks for val in adj_values):
                frontier = True
        if self.makes_2x2(origin, shape):
                return False # forma um quadrado
        if frontier:
            return True # faz parte da fronteira
        if not any(self.haspiece):
            return True # tabuleiro vazio, logo faz sempre parte da fronteira
        return False # não faz parte da fronteira

    def region_cells(self, region_id: int) -> list[tuple[int, int]]:
        """Devolve a lista de posições (row, col) pertencentes a uma região."""
        positions = list(zip(*np.where(self.board == region_id)))
        return positions
    
    def region_at(self, row: int, col: int) -> str:
        """Retorna a região original (número) da célula, mesmo que já esteja preenchida com uma peça."""
        for i, cells in enumerate(self.cells):
            if (row, col) in cells:
                return self.regions[i]
        return None
        
    def place(self, shape: list[tuple[int, int]], origin: tuple[int, int], mark) -> 'Board':
        """Retorna uma nova instância de Board com a peça colocada, marcada com `mark`."""
        new_board = np.copy(self.board)
        for dr, dc in shape:
            r, c = origin[0] + dr, origin[1] + dc
            new_board[r, c] = mark
        return Board(new_board, np.copy(self.haspiece), np.copy(self.regions), np.copy(self.cells))

    def adjacent_regions(self, region: int, diag: bool = False) -> list:
        """Devolve uma lista das regiões que fazem fronteira com a região enviada no argumento."""
        regions = set()
        region_cells = self.cells[int(region) - 1]

        for r, c in region_cells:
            adjacents = self.adjacent_positions(r, c, diag)
            for ar, ac in adjacents:
                adj_region = self.region_at(ar, ac)
                if adj_region and adj_region != region:
                    regions.add(adj_region)

        return list(regions)
    
    def adjacent_positions(self, r: int, c: int, diag: bool = False) -> list:
        """
        Retorna uma lista de posições adjacentes da posição (r, c) do tabuleiro.
        Se diag = True, consideramos as diagonais
        """
        adjacent = set()
        
        if diag:
            directions = [(-1,0),(1,0),(0,-1),(0,1)]
        else:
            directions = [  (-1, -1), (-1, 0), (-1, 1),
                            (0, -1),           (0, 1),
                            (1, -1),  (1, 0),  (1, 1)  ]
        
        if 0 <= r < self.height and 0 <= c < self.width:
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    adjacent.add((nr, nc))

        return list(adjacent)
    
    def adjacent_positions_shape(self, shape: list[tuple[int, int]], origin: tuple[int, int]) -> list:
        """Devolve as posições adjacentes à forma colocada no tabuleiro a partir de `origin`."""
        adjacent = set()
        for dr, dc in shape:
            r, c = origin[0] + dr, origin[1] + dc
            adjacent.update(self.adjacent_positions(r, c, diag=True))
        return sorted(adjacent)
    
    def adjacent_values(self, row:int, col:int, diag: bool) -> list:
        """Devolve os valores das celulas adjacentes à posição, em todas as direções, incluindo diagonais.
        Formato: [LeftTopDiag, , Top, RightTopDiag, Left, Right, , BottomLeftDiag, Bottom, BottomRightDiag]"""
        values = np.array([])

        if diag:
            directions = [  (-1, -1),   (-1, 0),    (-1, 1),
                            (0, -1),                (0, 1),
                            (1, -1),    (1, 0),     (1, 1)  ]
        else:
            directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]

        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.board.shape[0] and 0 <= c < self.board.shape[1]:
                values = np.append(values, self.board[r][c])

        return values
    
    def compute_actions(self) -> dict:
        """
        Gera todas as ações possíveis para o tabuleiro atual.
        """
        actions = defaultdict(list)
        for region in self.regions:
            if self.haspiece[int(region) - 1]:
                continue
            region_cells = self.cells[int(region) - 1]
            for origin in region_cells:
                for mark, shape_group in PIECES.items():
                    for shape in shape_group:
                        if self.can_place(shape, origin, region, mark):
                            actions[region].append((region, shape, origin, mark))
        return dict(actions)

    @staticmethod
    def parse_instance() -> 'Board':
        """
        Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.
        """
        board_list = []

        for line in stdin.readlines():
            line_ar = [elem for elem in line.split()]
            board_list.append(line_ar)
        board = np.array(board_list, dtype = '<U2')

        return Board(board, [], [], [])

class Nuruomino(Problem):
    def __init__(self, board: Board):
        super().__init__(NuruominoState(board))
        self.computed_actions = board.compute_actions()

    def actions(self, state: NuruominoState) -> list:
        """
        Gera todas as formas válidas de colocar uma peça no estado atual.
        """
        if not True in state.board.haspiece:
            actions = self.computed_actions
            sorted_actions = sorted(actions.items(), key=lambda x: len(x[1]))
            return [action for _, group in sorted_actions for action in group]
        else:
            filled = []
            for region in state.board.regions:
                if state.board.haspiece[int(region) - 1]:
                    filled.append(region)
            
            adjacent_regions = set()
            for region in filled:
                adjacent_regions.update(state.board.adjacent_regions(region))

            actions = []
            for neighbor in adjacent_regions:
                neighbor_actions = self.computed_actions[neighbor]
                for action in neighbor_actions:
                    region_id, shape, origin, mark = action
                    if state.board.can_place(shape, origin, region_id, mark):
                        actions.append((region_id, shape, origin, mark))
            
    
            # Obter ações agrupadas por região
            region_actions = defaultdict(list)
            for action in actions:
                region_id = action[0]
                region_actions[region_id].append(action)

            # Ordenar regiões pelo número de ações disponíveis
            sorted_groups = sorted(region_actions.items(), key=lambda x: len(x[1]))

            sorted_actions = [action for _, group in sorted_groups for action in group]
            return sorted_actions

    def result(self, state: NuruominoState, action) -> NuruominoState:
        """Retorna o estado resultante de executar a 'action' sobre 'state'."""
        region_id, shape, origin, mark = action
        new_board = state.board.place(shape, origin, mark)
        new_board.haspiece[int(region_id) - 1] = np.bool(True)
        return NuruominoState(new_board)
        

    def goal_test(self, state: NuruominoState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        if not(np.bool(False) in state.board.haspiece):
            return True
        else:
            return False

    def h(self, node: Node):
        """Retorna o nº de regiões por preencher no board"""
        regioes_a_preencher = sum(not haspiece for haspiece in node.state.board.haspiece)
        return regioes_a_preencher




if __name__ == "__main__":
    # Ler o tabuleiro do standard input e cria uma instância da classe Board
    problem_board = Board.parse_instance()

    #print(problem_board.haspiece)
    
    #print(problem_board)
    # Criar uma instância do problema
    problem = Nuruomino(problem_board)

    # Criar o estado inicial do problema
    initial_state = NuruominoState(problem_board)
    #print(initial_state.board.cells)
    #s1 = problem.result(initial_state, (np.str_('1'), frozenset({(1, 0), (0, 1), (2, 0), (0, 0)}), (np.int64(0), np.int64(0)), 'L'))
    #print(s1.board)
    #print(problem.actions(initial_state))
    #s2 = problem.result(s1,(np.str_('5'), frozenset({(1, 0), (2, 0), (0, 0), (3, 0)}), (np.int64(2), np.int64(5)), 'I'))
    #print(s2.board.haspiece)
    #s3 = problem.result(s2,(np.str_('4'), frozenset({(0, 1), (1, 0), (0, 2), (0, 0)}), (np.int64(4), np.int64(0)), 'L'))
    #s4 = problem.result(s3,(np.str_('2'), frozenset({(0, 1), (1, 0), (1, 1), (2, 1)}), (np.int64(0), np.int64(2)), 'T'))
    #s5 = problem.result(s4, (np.str_('3'), frozenset({(1, 0), (0, 1), (2, 0), (0, 0)}), (np.int64(0), np.int64(4)), 'L'))
    #print(s3.board)
    #print(s4.board)
    #print(s5.board)
    #print(s5.board.adjacent_values(1, 0))
    #print(problem.actions(s4))
    #print(problem.goal_test(s5))
    result = depth_first_graph_search(problem)
    if result is not None:
        result = result.state
        print(result.board)
    else:
        print("No solution found.")