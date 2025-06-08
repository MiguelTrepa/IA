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
        """ 
        Este método é utilizado em caso de empate na gestão da lista
        de abertos nas procuras informadas. 
        """
        return self.id < other.id
    
    def __hash__(self):
        """
        Este método é usado para calcular o hash do estado.
        Usado para evitar estados repetidos na procura e acelarar comparações.
        """
        if self._hash is None:
            # Calcula o hash baseado no haspiece e nos bytes do tabuleiro
            self._hash = hash((tuple(self.board.haspiece), self.board.board.tobytes()))
        return self._hash

    def __eq__(self, other: object) -> bool:
        """
        Método de comparação de igualdade entre dois estados.
        Compara os haspieces e os bytes do tabuleiro para verificar se são iguais.
        """
        # Solução instantânea
        if self is other:
            return True

        # No caso de já termos calculado os hashes, podemos comparar diretamente
        if getattr(self, "_hash", None) is not None \
        and getattr(other, "_hash", None) is not None \
        and self._hash != other._hash:
            return False

        # Verifica se os haspieces são iguais
        if self.board.haspiece != other.board.haspiece:
            return False
        
        # Assume que os tabuleiros são iguais se os haspieces são iguais
        return True
    
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
        """
        Devolve uma representação textual do tabuleiro.
        """
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

                    if not (0 <= piece_r < height - 1 and 0 <= piece_c < width - 1): continue

                    # Conta quantas posições do quadrado estão preenchidas
                    filled = 0
                    if (piece_r, piece_c) in piece or board[piece_r, piece_c] in MARKS:
                        filled += 1
                    if (piece_r, piece_c + 1) in piece or board[piece_r, piece_c + 1] in MARKS:
                        filled += 1
                    if (piece_r + 1, piece_c) in piece or board[piece_r + 1, piece_c] in MARKS:
                        filled += 1
                    if (piece_r + 1, piece_c + 1) in piece or board[piece_r + 1, piece_c + 1] in MARKS:
                        filled += 1
                    
                    # Se todas as posições do quadrado estão preenchidas
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
        Verifica se é possível colocar a forma no tabuleiro a partir de origin, de acordo com a regras. Não precisa de verificar se uma peça cabe na região
        """
        # Verifica se a região já está preenchida
        id = region_id - 1
        if id < 0 or id >= len(self.haspiece) or self.haspiece[id]:
            return False
        
        # Carrega os dados do tabuleiro
        board = self.board
        height, width = self.height, self.width
        fronter = False

        # Verifica as adjacências da peça
        for sr, rc in shape:
            r, c = origin[0] + sr, origin[1] + rc
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                value = board[nr, nc]
                if value == mark:
                    return False    # A peça tocaria numa peça do mesmo tipo
                if value in MARKS:
                    fronter = True  # A peça toca numa peça diferente
        
        # Se não for a primeira peça e não tocar numa peça
        if not fronter and any(self.haspiece):
            return False

        # Vê se a peça faz um O (quadrado 2x2)
        if self.makes_2x2(origin, shape):
            return False
        return True
    
    def place(self, shape: list[tuple[int, int]], origin: tuple[int, int], mark) -> 'Board':
        """
        Retorna uma nova instância de Board com a peça colocada, marcada com mark.
        """
        new_board_array = self.board.copy()
        for dr, dc in shape:
            new_board_array[origin[0] + dr, origin[1] + dc] = mark
        
        new_haspiece = self.haspiece.copy()
        
        region_id = self.board[origin[0], origin[1]]
        new_haspiece[region_id - 1] = True

        return Board(new_board_array, new_haspiece, self.regions, self.cells)

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
                # Divide a linha em números inteiros
                lines.append([int(r) for r in line.split()])

        board_data = np.array(lines, dtype=object)
        
        return Board(board_data)

class Nuruomino(Problem):
    def __init__(self, board: Board):
        super().__init__(NuruominoState(board))
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
            
            # Ordena as ações pela prioridade da peça
            region_actions.sort(key=lambda x: self.action_priority[x[3]])
            potential_actions[region_id] = region_actions

        return potential_actions

    def actions(self, state: NuruominoState):
        """
        Retorna uma lista de ações possíveis a partir do estado atual.
        Experiência mostra que devolver as ações das regiões com o segundo   ou terceiro menor número de ações é o melhor equilibrio entre velocidade e exploração.
        """
        # Carrega os dados do estado para evitar cálculos repetidos
        board = state.board
        can_place = board.can_place
        potential = self.potential_actions
        regions = board.regions
        haspiece = board.haspiece
        cells = board.cells

        # Faz a lista de regiões a verificar
        empty = [rid for rid in regions if not haspiece[rid-1]]

        # Calcula as ações possíveis para cada região
        region_actions = {}
        counts = set()
        for rid in empty:
            # Região forçada
            if len(cells[rid]) == 4:
                return potential[rid]

            # Encontrar ações possíveis
            acts = [
                a for a in potential[rid]
                if can_place(a[0], a[1], a[2], a[3])
            ]
            if acts:
                region_actions[rid] = acts
                counts.add(len(acts))   # Número usado para filtrar ações

        # Se não houver ações, não acrescentamos nada à fronteira
        if not counts:
            return []

        # Encontra o terceiro menor número de ações
        inf = float('inf')
        first = second = third = inf
        for x in counts:
            if x < first:
                third, second, first = second, first, x
            elif first < x < second:
                third, second = second, x
            elif second < x < third:
                third = x
        third_min = third if third < inf else first

        # Filtra as ações das regiões com menos que o terceiro menor número de  ações
        small = []
        # Filtra as ações das regiões com o terceiro menor número de ações
        eq    = []
        # Filtra as ações das regiões com apenas uma ação
        single = []
        for acts in region_actions.values():
            l = len(acts)
            if l == 1: single.append(acts[0])
            if   l <  third_min: small.extend(acts)
            elif l == third_min: eq.extend(acts)
        if single:
            return single
        return small or eq


    def result(self, state: NuruominoState, action) -> NuruominoState:
        """
        Retorna o estado resultante de executar a 'action' sobre 'state'.
        A ação é um tuplo (shape, origin, region_id, mark)
        """
        new_board = state.board.place(action[0], action[1], action[3])
        return NuruominoState(new_board)
        

    def goal_test(self, state: NuruominoState):
        """
        Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema.
        Neste caso, o estado objetivo é alcançado quando todas as regiões
        do tabuleiro têm uma peça colocada.
        """
        return all(state.board.haspiece)

    def h(self, node: Node):
        """
        Retorna o nº de regiões por preencher no board
        Como usamos DFS para procurar, a heurística não é usada
        """
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