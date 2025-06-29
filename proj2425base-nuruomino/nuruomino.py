# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 18:
# 109370 Miguel Trêpa
#  90173 Rafael Ferreira


from sys import stdin
import numpy as np
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
# Direções adjacentes com diagonal
DIAGONAL_DIRECTIONS = ((-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1))

class NuruominoState:
    state_id = 0

    def __init__(self, board: 'Board'):
        self.board = board
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1
        self._hash = None # Guardamos a hash para maior eficiência na procura (evitar estados repetidos)

    def __lt__(self, other: 'NuruominoState') -> bool:
        """ Este método é utilizado em caso de empate na gestão da lista
        de abertos nas procuras informadas. """
        return self.id < other.id
    
    def __hash__(self):
        """ Retorna um hash único para o estado, baseado no conteúdo do tabuleiro. """
        if self._hash is None:
            # Combina o tabuleiro e as regiões preenchidas num único hash
            self._hash = hash((self.board.board.tobytes(), tuple(self.board.haspiece)))
        return self._hash
    
    def __eq__(self, other: 'NuruominoState') -> bool:
        """ 
        Verifica se dois estados são iguais, comparando o conteúdo do tabuleiro e as regiões preenchidas. 
        """
        if not isinstance(other, NuruominoState):
            return False
        # Compara o tabuleiro e as regiões preenchidas
        return (np.array_equal(self.board.board, other.board.board) and 
                self.board.haspiece == other.board.haspiece)

class Board:
    """
    Representação interna de um tabuleiro do Puzzle Nuruomino.

    Args:
        board (np.ndarray): A matriz com identificadores de regiões.
        haspiece (list[bool]): Flags indicando se cada região já tem uma peça.
        regions (list[str]): Lista das regiões presentes.
    """
    def __init__(self, board=None, haspiece=None, regions=None, cells=None):
        if board is None:
            board = np.array([[1]], dtype=object)

        self.board = board
        
        self.height, self.width = self.board.shape

        # Se for a primeira vez que é criado o tabuleiro
        if haspiece is None:
            # Memoriza as regiões do tabuleiro
            unique_regions = np.unique(board)
            unique_regions_int = sorted([int(ur) for ur in unique_regions])
            self.regions = unique_regions_int

            # Criamos um bool para cada região que indica se já tem uma peça colocada
            region_count = len(self.regions)
            self.haspiece = [False] * region_count

            # Criamos uma lista de listas que guarda as coordenadas de cada região
            self.cells = [[] for _ in range(region_count)]
            for r in range(self.height):
                for c in range(self.width):
                    self.cells[int(self.board[r, c]) - 1].append((r, c))
            
            # Converte o cells de lista de listas para tuplo de tuplos
            # Para ser imutável
            self.cells = tuple(tuple(l) for l in self.cells)
        else:
            # Se já houver peças colocadas
            self.haspiece = list(haspiece)
            self.cells = cells
            self.regions = regions

    def __str__(self):
        """Devolve uma representação textual do tabuleiro."""
        return '\n'.join('\t'.join(str(cell) for cell in row) for row in self.board)
    
    def makes_2x2(self, piece_coordinates: set[tuple[int, int]]) -> bool:
        """ Verifica se a peça forma um quadrado 2x2"""
        height, width = self.board.shape
        
        corners = set()
        for r, c in piece_coordinates:
            for dr, dc in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
                top_left_corner_r, top_left_corner_c = r + dr, c + dc
                # Verifica que os cantos estão no tabuleiro
                if 0 <= top_left_corner_r < height - 1 and 0 <= top_left_corner_c < width - 1:
                    corners.add((top_left_corner_r, top_left_corner_c))
        
        for top_left_corner_r, top_left_corner_c in corners:
            square_cells = [
                (top_left_corner_r, top_left_corner_c), (top_left_corner_r, top_left_corner_c + 1),
                (top_left_corner_r + 1, top_left_corner_c), (top_left_corner_r + 1, top_left_corner_c + 1)
            ]
            
            filled_count = 0
            for sq_r, sq_c in square_cells:
                if ((sq_r, sq_c) in piece_coordinates or 
                    (isinstance(self.board[sq_r, sq_c], str) and self.board[sq_r, sq_c] in MARKS)):
                    filled_count += 1 # Tem um marca da peça
            
            if filled_count == 4: # Forma quadrao
                return True
        
        return False
    
    def can_place(self, shape: list[tuple[int, int]], origin: tuple[int, int], region_id: int, mark: str) -> bool:
        """ Verifica se é possível colocar uma peça"""

        region_index = int(region_id) - 1
        if region_index < 0 or region_index >= len(self.haspiece) or self.haspiece[region_index]: # região já tem peça
            return False
        
        height, width = self.board.shape
        current_region_cells = set(self.cells[region_index])
        is_first_piece = not any(self.haspiece)
        piece_positions = []
        frontier = False
        directions = DIRECTIONS
        
        for dr, dc in shape:
            r, c = origin[0] + dr, origin[1] + dc
            if not (0 <= r < height and 0 <= c < width and (r, c) in current_region_cells): # Fora do tabuleiro
                return False
            if not frontier or not is_first_piece: 
                for adjacent_dr, adjacent_dc in directions:
                    adjacent_r, adjacent_c = r + adjacent_dr, c + adjacent_dc
                    if 0 <= adjacent_r < height and 0 <= adjacent_c < width:
                        adjacent_cell_value = self.board[adjacent_r, adjacent_c]
                        if adjacent_cell_value == mark: # Tem peça adjacente igual
                            return False
                        if not is_first_piece and not frontier and adjacent_cell_value in MARKS:
                            frontier = True
            piece_positions.append((r, c))

        if not is_first_piece and not frontier: # Não faz parte da fronteira
            return False
        
        if self.makes_2x2(set(piece_positions)): # Forma uma quadrado
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

        adjacent_cells = self.adjacent_positions(*region_cells[0], diag)

        for r, c in adjacent_cells:
            if self.board[r, c] != region:
                regions.add(self.board[r, c])
        
        return list(regions)
    
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
            
            region_actions.sort(key=lambda x: self.action_priority[x[3]])
            potential_actions[region_id] = region_actions

        return potential_actions    

    def actions(self, state: NuruominoState):
        """Gera todas as formas válidas de colocar uma peça no estado atual."""
        board = state.board

        empty_regions_ids = [
            region_id for region_id in board.regions if not board.haspiece[region_id - 1]
        ]
        for region_id in empty_regions_ids:
            inference_region_cells = board.cells[int(region_id) - 1]
            if len(inference_region_cells) == 4:
                for action in self.potential_actions[region_id]:
                    if board.can_place(action[1], action[2], action[0], action[3]):
                        return [action]
                    
        actions_by_region_size = []
        for region_id in empty_regions_ids: # Não fez inferências
            region_size = len(board.cells[region_id - 1])
            region_actions = []
            for action in self.potential_actions[region_id]:
                if board.can_place(action[1], action[2], action[0], action[3]):
                    region_actions.append(action)
            if region_actions:
                actions_by_region_size.append((region_size, region_actions))
        
        actions_by_region_size.sort(key=lambda x: x[0]) # Sort pelo tamanho da região

        actions = []
        for _, actions_list in actions_by_region_size:
            actions.extend(actions_list) # Dá flatten à lista
            
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
        # print(self.cnt)
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