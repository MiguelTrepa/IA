# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 18:
# 109370 Miguel Trêpa
#  90173 Rafael Ferreira


from collections import defaultdict
from sys import stdin
import numpy as np
from search import Problem, Node, depth_first_graph_search

# Tetromino definitions (with all rotations and reflections)
PIECES = {
    'L': [
        [(0,0), (1,0), (2,0), (2,1)], 
        [(0,0), (0,1), (0,2), (1,0)],
        [(0,0), (0,1), (1,1), (2,1)], 
        [(0,2), (1,0), (1,1), (1,2)],
        [(0,1), (1,1), (2,1), (2,0)], 
        [(1,0), (1,1), (1,2), (0,2)],
        [(0,0), (0,1), (1,0), (2,0)], 
        [(0,0), (1,0), (1,1), (1,2)]
    ],
    'I': [
        [(0,0), (1,0), (2,0), (3,0)], 
        [(0,0), (0,1), (0,2), (0,3)]
    ],
    'T': [
        [(-1, 0), (0, -1), (0, 0), (0, 1)],
        [(0, -1), (0, 0), (1, 0), (-1, 0)],
        [(1, 0), (0, -1), (0, 0), (0, 1)],
        [(0, 1), (0, 0), (1, 0), (-1, 0)]
    ],
    'S': [
        [(0,1), (0,2), (1,0), (1,1)],
        [(0,0), (1,0), (1,1), (2,1)],
        [(0,0), (0,1), (1,1), (1,2)],
        [(0,1), (1,0), (1,1), (2,0)]   
    ]
}

class Board:
    """
    Representação interna de um tabuleiro do Puzzle Nuruomino.
    """
    def __init__(self, board = None,  haspiece = None, connected_regions = None):
        if board is None:
            board = Board.parse_instance()
        self.board = board
        self.height, self.width = self.board.shape
        self.regions = self._extract_regions()
        self.neighbors = self._find_neighbors()
        if haspiece is None:
            self.haspiece = np.zeros(len(self.regions), dtype=bool)
        else:
            self.haspiece = haspiece
        self.connected_regions = connected_regions if connected_regions is not None else []
    
    def _extract_regions(self):
        """
        Extrai as regiões do tabuleiro e retorna um dicionário
        onde as chaves são os símbolos das regiões e os valores são
        listas de coordenadas (tuplas) que pertencem a cada região.
        """
        regions = defaultdict(list)
        for r in range(self.height):
            for c in range(self.width):
                regions[self.board[r, c]].append((r, c))
        return dict(regions)

    def _find_neighbors(self):
        neighbors = defaultdict(set)
        for r in range(self.height):
            for c in range(self.width):
                reg = self.board[r, c]
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        nreg = self.board[nr, nc]    
                        if nreg != reg:
                            neighbors[reg].add(nreg)
        return dict(neighbors)
    
    @staticmethod
    def parse_instance() -> 'Board':
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.
        """
        board_list = []

        for line in stdin.readlines():
            line_ar = [elem for elem in line.split()]
            board_list.append(line_ar)
        board = np.array(board_list, dtype = 'U1')

        return Board(board)

    def __str__(self):
        """Devolve uma representação textual do tabuleiro."""
        formatted_board = ""
        for row in self.board:
            formatted_board += "\t".join(str(cell) for cell in row) + "\n"
        return formatted_board
    
    def region_at(self, r: int, c: int) -> str:
        """
        Retorna o símbolo da região na posição (r, c) do tabuleiro.
        """
        if 0 <= r < self.height and 0 <= c < self.width:
            key = [key for key, value in self.regions.items() if (r, c) in value]
            return key[0]
        return None
    
    def group_at(self, region: str) -> list:
        """
        Retorna a lista de posições (tuplas) que pertencem à região especificada.
        """
        for group in self.connected_regions:
            if region in group:
                return group
        return None
    
    def connect_groups(self, region_a: str, region_b: str):
        """
        Conecta duas regiões no dicionário de regiões conectadas.
        """
        group_a = self.group_at(region_a)
        group_b = self.group_at(region_b)

        if group_a and group_b:
            if group_a != group_b:
                group_a.update(group_b)
                self.connected_regions.remove(group_b)
        elif group_a:
            group_a.add(region_b)
        elif group_b:
            group_b.add(region_a)
        else:
            self.connected_regions.append({region_a, region_b})
    
    def adjacent_positions(self, r: int, c: int, diag = False) -> list:
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
    
    def region_adjacent_positions(self, r: int, c: int, diag = False) -> list:
        """
        Retorna uma lista de posições adjacentes (cima, baixo, esquerda, direita) da região na posição (r, c) do tabuleiro.
        """
        adjacent = set()
        region = self.region_at(r, c)

        for row, col in self.regions[region]:
            adjacent_positions = self.adjacent_positions(row, col, diag)
            for pos in adjacent_positions:
                if region != self.region_at(pos[0], pos[1]):
                    adjacent.add(pos)

        return list(adjacent)
    
    def adjacent_values(self, r: int, c: int, diag = False) -> set:
        """
        Retorna o conjunto de valores adjacentes à posição (r, c) do tabuleiro.
        """
        adjacent = set()

        if diag:
            directions = [(-1,0),(1,0),(0,-1),(0,1)]
        else:
            directions = [  (-1, -1), (-1, 0), (-1, 1),
                            (0, -1),           (0, 1),
                            (1, -1),  (1, 0),  (1, 1)   ]

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                value = self.board[nr, nc]
                if value != self.board[r, c]:
                    adjacent.add(value)
        return adjacent

    def piece_adjacent_values(self, origin: tuple[int, int], shape: list[tuple[int, int]], diag = False) -> set:
        """
        Retorna um conjunto de símbolos das regiões adjacentes às posições ocupadas pela peça.
        A peça é representada por uma lista de tuplas (r, c) que indicam as posições ocupadas.
        """
        adjacent = set()
        for r, c in shape:
            nr, nc = origin[0] + r, origin[1] + c
            values = self.adjacent_values(nr, nc, diag)
            adjacent.update(values)
        return adjacent
    
    def is_region_filled(self, region_label: str) -> bool:
        """
        Verifica se a região já está preenchida com uma peça ('L', 'I', 'T', 'S').
        Retorna True se qualquer célula da região tiver um desses valores.
        """
        marks = {'L', 'I', 'T', 'S'}
        for r, c in self.regions[region_label]:
            if self.board[r, c] in marks:
                return True
        return False

    def fits(self, origin:tuple[int, int], shape: list[tuple[int, int]]) -> bool:
        """
        Testa se a peça pode ser colocada no tabuleiro na região especificada por origin.
        """
        placed = [(origin[0] + dx, origin[1] + dy) for dx, dy in shape]
        if all(cell in board.regions[board.region_at(origin[0], origin[1])] for cell in placed):
            return True
        return False
    
    def has_border(self, origin: tuple[int, int], shape: list[tuple[int, int]]) -> bool:
        """
        Verifica se a peça tem uma fronteira com outra região.
        Retorna True se a peça tiver uma fronteira com outra região.
        """
        region = self.region_at(origin[0], origin[1])
        adjacent_labels = self.piece_adjacent_positions(origin, shape)
        if any(self.region_at(r, c) != region for r, c in adjacent_labels):
            return True
        return False

    def equal_adjacent(self, origin: tuple[int, int], shape: list[tuple[int, int]], piece_label: str) -> bool:
        """
        Verifica se uma peça pode ser colocada sem ter uma peça igual adjacente a ela.
        Retorna True se a peça não puder ser colocada
        """
        adjacent_labels = self.piece_adjacent_values(origin, shape)
        return piece_label in adjacent_labels
    
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
            temp_board[r, c] = 'L'

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
    
    def is_valid(self, origin: tuple[int, int], shape: list[tuple[int, int]], piece_label: str) -> bool:
        """
        Verifica se a peça pode ser colocada na posição origin da região.
        """
        # Verifica se a região está preenchida
        region_label = self.region_at(origin[0], origin[1])
        if self.is_region_filled(region_label):
            return False
        # Verifica se a peça cabe na região
        if not self.fits(origin, shape):
            return False
        # Verifica se a peça tem uma fronteira com outra região
        if not self.has_border(origin, shape):
            return False
        # Verifica se a peça não tem uma peça igual adjacente
        if self.equal_adjacent(origin, shape, piece_label):
            return False
        # Verifica se a colocação da peça não forma um quadrado 2x2
        if self.makes_2x2(origin, shape):
            return False
        return True
    
    def all_regions_connected(self):
        return any(len(group) == len(self.regions) for group in self.connected_regions)

    def place(self, shape: list[tuple[int, int]], origin: tuple[int, int], mark, region) -> 'Board':
        """Retorna uma nova instância de Board com a peça colocada, marcada com `mark`."""
        new_board = np.copy(self.board)
        new_haspiece = np.copy(self.haspiece)
        region = self.region_at(origin[0], origin[1])
        new_haspiece[np.int_(region) - 1] = True

        for dr, dc in shape:
            r, c = origin[0] + dr, origin[1] + dc
            new_board[r, c] = mark

        adjacent_positions = self.piece_adjacent_positions(origin, shape)
        for pos in adjacent_positions:
            if self.board[pos[0], pos[1]] in {'L', 'I', 'T', 'S'}:
                neighbor_region = self.region_at(pos[0], pos[1])
                if neighbor_region != region:
                    self.connect_groups(region, neighbor_region)

        return Board(new_board, new_haspiece)
    
    def remove(self, shape: list[tuple[int, int]], origin: tuple[int, int]) -> 'Board':
        """Retorna uma nova instância de Board com a peça removida da posição origin."""
        new_board = np.copy(self.board)
        for dr, dc in shape:
            r, c = origin[0] + dr, origin[1] + dc
            new_board[r, c] = self.region_at(r, c)

        # Remove as conexões da região
        region = self.region_at(origin[0], origin[1])
        if region in self.connected_regions:
            for neighbor in self.connected_regions[region]:
                if neighbor in self.connected_regions:
                    self.connected_regions[neighbor].discard(region)
            del self.connected_regions[region]

class CSP:
    def __init__(self, board: Board):
        self.board = board
        self.variables = list(board.regions.keys())
        self.domains = self.compute_domains()

    def compute_domains(self):
        return {region: self._generate_options(region, cells) for region, cells in self.board.regions.items()}

    def _generate_options(self, region, cells):
        options = []
        for origin in cells:
            options.extend(self._validate_piece_placement(origin))
        return options

    def _validate_piece_placement(self, origin):
        valid_options = []
        for piece, shapes in PIECES.items():
            for shape in shapes:
                if self.board.is_valid(origin, shape, piece):
                    valid_options.append((piece, shape, origin))
        return valid_options


class NuruominoState:
    state_id = 0

    def __init__(self, board: 'Board'):
        self.board = board
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1

    def __lt__(self, other: 'NuruominoState') -> bool:
        return self.id < other.id
    
    def __eq__(self, other):
        if not isinstance(other, NuruominoState):
            return False
        return np.array_equal(self.board.board, other.board.board)
    
    def __hash__(self):
        return hash(self.board.board.tobytes())

class Nuruomino(Problem):
    def __init__(self, board: Board, csp: CSP):
        super().__init__(NuruominoState(board))
        self.csp = csp

    def actions(self, state: NuruominoState):
        actions = []
        board = state.board
        for region, cells in board.regions.items():
            if board.is_region_filled(region):
                continue
            for origin in cells:
                for piece, shapes in PIECES.items():
                    for shape in shapes:
                        if board.is_valid(origin, shape, piece):
                            actions.append((region, (piece, shape, origin)))
        return actions

    def result(self, state: NuruominoState, action) -> NuruominoState:
        region_id, (mark, shape, origin) = action
        new_board = state.board.place(shape, origin, mark, region_id)
        return NuruominoState(new_board)

    def goal_test(self, state: NuruominoState):
        all_filled = all(board.is_region_filled(region) for region in board.regions)
        return all_filled

if __name__ == "__main__":
    # Exemplo de uso
    board = Board.parse_instance()
    print("Tabuleiro lido:")
    print(board)

    csp = CSP(board)
    print("Regiões extraídas:")
    for region, cells in csp.board.regions.items():
        print(f"{region}: {cells}")
    
    print("Vizinhos de cada região:")
    for region, neighbors in csp.board.neighbors.items():
        print(f"{region}: {neighbors}")

    print("Domínios das regiões:")
    for region, options in csp.domains.items():
        print(f"{region}: {options}")
        print(f"  Opções válidas: {len(options)}")
    
    # Criação do estado inicial
    initial_state = NuruominoState(board)
    
    # Exemplo de criação do problema
    problem = Nuruomino(board, csp)
    
    # Exemplo de busca (ainda não implementada)
    solution = depth_first_graph_search(problem)
    print("Solução encontrada:", solution)