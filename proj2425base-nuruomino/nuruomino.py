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

# Sets das formas únicas para melhorar procura 
# Apenas usado para verificar se uma peça faz um quadrado 2x2
SHAPE_SETS = {id(shape): frozenset(shape)
              for shapes in PIECES.values() for shape in shapes}

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
            exit("Tabuleiro não fornecido. Certifique-se de que o input está correto.")

        self.board = board
        
        self.height, self.width = self.board.shape

        # Se for a primeira vez que é criado o tabuleiro
        if haspiece is None:
            # Memoriza as regiões do tabuleiro
            unique_regions_int = sorted(region for region in np.unique(board))
            self.regions = unique_regions_int

            # Criamos um bool para cada região que indica se já tem uma peça colocada
            region_count = len(self.regions)
            self.haspiece = [False] * region_count

            # Criamos uma lista de listas que guarda as coordenadas de cada região
            region_cells = {rid: set() for rid in unique_regions_int}
            for r in range(self.height):
                for c in range(self.width):
                    region_cells[self.board[r, c]].add((r, c))
            self.cells = region_cells
        else:
            # Se já houver peças colocadas
            self.haspiece = list(haspiece)
            self.cells = cells
            self.regions = regions

    def __str__(self):
        """Devolve uma representação textual do tabuleiro."""
        return '\n'.join('\t'.join(str(cell) for cell in row) for row in self.board)
    
    def region_at(self, r: int, c: int) -> int:
        """
        Retorna o ID da região na posição (r, c) do tabuleiro.
        Se a posição estiver fora dos limites, retorna None.
        """
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.board[r, c]
        return None
    
    
    def makes_2x2(self, origin: tuple[int, int], piece: list[tuple[int, int]]) -> bool:
        """
        Verifica se a colocação de uma peça formaria um quadrado 2x2
        """
        # Usamos o set de peças para procura mais rápida
        piece = SHAPE_SETS[id(piece)]
        r0, c0 = origin
        # Ajusta as coordenadas da peça
        piece = {(r + r0, c + c0) for r, c in piece}
        board, width, height = self.board, self.width, self.height

        # Vê a região que rodeia a peça
        for r, c in piece:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    piece_r, piece_c = r + dr, c + dc

                    if not (0 <= piece_r < height - 1 and 0 <= piece_c < width - 1):
                        continue

                    filled = 0

                    if (piece_r, piece_c) in piece or board[piece_r, piece_c] in MARKS:
                        filled += 1
                    if (piece_r, piece_c + 1) in piece or board[piece_r, piece_c + 1] in MARKS:
                        filled += 1
                    if (piece_r + 1, piece_c) in piece or board[piece_r + 1, piece_c] in MARKS:
                        filled += 1
                    if (piece_r + 1, piece_c + 1) in piece or board[piece_r + 1, piece_c + 1] in MARKS:
                        filled += 1
                    
                    if filled == 4:
                        return True
        return False
    
    def fits(self, shape: list[tuple[int, int]], origin: tuple[int, int]) -> bool:
        """
        Verifica se a forma pode ser colocada no tabuleiro a partir de origin.
        Não verifica se a região já está preenchida.
        """
        region = self.region_at(origin[0], origin[1])
        for dr, dc in shape:
            r, c = origin[0] + dr, origin[1] + dc
            
            # Verifica se a posição está dentro dos limites do tabuleiro
            if not (0 <= r < self.height and 0 <= c < self.width):
                return False
            
            value = self.board[r, c]
            if not value == region:
                return False
        return True
    
    def can_place(self, shape: list[tuple[int, int]], origin: tuple[int, int], region_id: int, mark: str) -> bool:
        """
        Verifica se é possível colocar a forma no tabuleiro a partir de origin, de acordo com a regras
        """
        # Verifica se a região já está preenchida
        region_index = region_id - 1
        if region_index < 0 or region_index >= len(self.haspiece) or self.haspiece[region_index]:
            return False
        
        # Fronteira declara que se a peça está em contacto com uma peça já colocada
        fronter = False

        for dr, dc in shape:
            r, c = origin[0] + dr, origin[1] + dc

            # Garante que a peça não estaria em contacto com uma peça igual
            for adjacent_r, adjacent_c in DIRECTIONS:
                adj_r, adj_c = r + adjacent_r, c + adjacent_c
                if (0 <= adj_r < self.height and 0 <= adj_c < self.width and 
                    self.board[adj_r, adj_c] == mark):
                    return False
                
            # Garante que a peça está em contacto com pelo menos uma peça já colocada
            if not fronter: 
                for adjacent_r, adjacent_c in DIRECTIONS:
                    adj_r, adj_c = r + adjacent_r, c + adjacent_c
                    if (0 <= adj_r < self.height and 0 <= adj_c < self.width and
                        self.board[adj_r, adj_c] in MARKS):
                        fronter = True
                        break
        
        # Vê se a peça faz um O (2x2 square)
        if self.makes_2x2(origin, shape):
            return False
        
        # Devolve se a peça pode ser colocada
        # Se a peça é a primeira, não é preciso que faça fronteira
        return fronter or not any(self.haspiece)
    
    def place(self, shape: list[tuple[int, int]], origin: tuple[int, int], mark) -> 'Board':
        """Retorna uma nova instância de Board com a peça colocada, marcada com `mark`."""
        new_board_array = self.board.copy()
        for dr, dc in shape:
            new_board_array[origin[0] + dr, origin[1] + dc] = mark
        
        new_haspiece = self.haspiece.copy()
        
        region_id = self.board[origin[0], origin[1]]
        new_haspiece[region_id - 1] = True

        return Board(new_board_array, new_haspiece, self.regions, self.cells)

    # def adjacent_regions(self, region:int, diag: bool) -> list:
    #     """Devolve uma lista das regiões que fazem fronteira com a região enviada no argumento."""
    #     # regions = set()
    #     # region_cells = self.cells[region]

    #     # adjacent_cells = self.adjacent_positions(*next(iter(region_cells)), diag)

    #     # for r, c in adjacent_cells:
    #     #     if self.board[r, c] != region:
    #     #         regions.add(self.board[r, c])
        
    #     # return list(regions)
    #     pass

    # def adjacent_positions(self, r: int, c: int, diag = False) -> list:
    #     """
    #     Retorna uma lista de posições adjacentes da posição (r, c) do tabuleiro.
    #     Se diag = True, consideramos as diagonais
    #     """
    #     # adjacent = set()
        
    #     # if diag:
    #     #     directions = [(-1,0),(1,0),(0,-1),(0,1)]
    #     # else:
    #     #     directions = [  (-1, -1), (-1, 0), (-1, 1),
    #     #                     (0, -1),           (0, 1),
    #     #                     (1, -1),  (1, 0),  (1, 1)  ]
        
    #     # if 0 <= r < self.height and 0 <= c < self.width:
    #     #     for dr, dc in directions:
    #     #         nr, nc = r + dr, c + dc
    #     #         if 0 <= nr < self.height and 0 <= nc < self.width:
    #     #             adjacent.add((nr, nc))

    #     # return list(adjacent)
    #     pass

    # def adjacent_values(self, row:int, col:int, diag: bool) -> list:
    #     """
    #     Devolve os valores das celulas adjacentes à posição, em todas as direções, incluindo diagonais.
    #     Formato: [LeftTopDiag, , Top, RightTopDiag, Left, Right, , BottomLeftDiag, Bottom, BottomRightDiag]
    #     """
    #     # Função não usada
    #     # values = np.array([])

    #     # if diag:
    #     #     directions = [  (-1, -1),   (-1, 0),    (-1, 1),
    #     #                     (0, -1),                (0, 1),
    #     #                     (1, -1),    (1, 0),     (1, 1)  ]
    #     # else:
    #     #     directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    #     # for dr, dc in directions:
    #     #     r, c = row + dr, col + dc
    #     #     if 0 <= r < self.board.shape[0] and 0 <= c < self.board.shape[1]:
    #     #         values = np.append(values, self.board[r][c])

    #     # return values
    #     pass

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
                lines.append([int(r) for r in line.split()])

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
            region_cells = board.cells[region_id]

            region_actions = []
            for origin in region_cells:
                for mark, shapes in PIECES.items():
                    for shape in shapes:
                        if board.fits(shape, origin):
                            # Adiciona a ação (shape, origin, region_id, mark)
                            region_actions.append((shape, origin, region_id, mark))
            
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
            inference_region_cells = board.cells[region_id]
            if len(inference_region_cells) == 4:
                for action in self.potential_actions[region_id]:
                    if board.can_place(action[0], action[1], action[2], action[3]):
                        return [action]
                    
        actions_by_region_size = []
        for region_id in empty_regions_ids: # Não fez inferências
            region_size = len(board.cells[region_id])
            region_actions = []
            for action in self.potential_actions[region_id]:
                if board.can_place(action[0], action[1], action[2], action[3]):
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
        new_board = state.board.place(action[0], action[1], action[3])
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