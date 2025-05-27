# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 18:
# 109370 Miguel Trêpa
#  90173 Rafael Ferreira


from sys import stdin
import numpy as np
from search import Problem, Node

# Each shape is a list of (row, col) offsets from the origin (0, 0)
L_SHAPE = [(0,0), (1,0), (2,0), (2,1)]

I_SHAPE = [(0,0), (1,0), (2,0), (3,0)]

T_SHAPE = [(0,0), (0,1), (0,2), (1,1)]

S_SHAPE = [(0,1), (0,2), (1,0), (1,1)]

# Rotates a piece by 90 degrees
def rotate_coords(coords):
    return [(y, -x) for x, y in coords]

# Returns a canonical coordinates representation of a piece as a frozen set
def canonical_coords(coords):
    x_min = min(x for x, _ in coords)
    y_min = min(y for _, y in coords)
    return frozenset((y - y_min, x - x_min) for x, y in coords)

# Makes all possible variations of a piece (optionally including reflections)
# as a set of canonical representations
def make_piece_variations(piece, reflections=True):
    variations = {canonical_coords(piece)}
    for i in range(3):
        piece = rotate_coords(piece)
        variations.add(canonical_coords(piece))
    if reflections:
        piece_reflected = [(y, x) for x, y in piece]
        variations.update(make_piece_variations(piece_reflected, False))
    return variations

# Generate all possible shapes
L_SHAPES = make_piece_variations(L_SHAPE)
I_SHAPES = make_piece_variations(I_SHAPE)
T_SHAPES = make_piece_variations(T_SHAPE)
S_SHAPES = make_piece_variations(S_SHAPE)


from sys import stdin
import numpy as np
from search import Problem, Node

class NuruominoState:
    state_id = 0

    def __init__(self, board: 'Board'):
        self.board = board
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1

    def __lt__(self, other):
        """ Este método é utilizado em caso de empate na gestão da lista
        de abertos nas procuras informadas. """
        return self.id < other.id

class Board:
    """Representação interna de um tabuleiro do Puzzle Nuruomino."""
    def __init__(self, board=np.array([1, 1], np.int8)):
        """Construtor da classe Board. Se o argumento for None, cria um tabuleiro vazio."""
        self.board = board
    
    def __str__(self):
        """Devolve uma representação textual do tabuleiro."""
        formatted_board = ""
        for row in self.board:
            formatted_board += "\t".join(str(cell) for cell in row) + "\n"
        return formatted_board
    
    def can_place(self, shape: list[tuple[int, int]], origin: tuple[int, int], region_id: int, mark: str) -> bool:
        """Verifica se é possível colocar a forma no tabuleiro a partir de `origin`, respeitando os limites da região."""
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
        return True

    def region_cells(self, region_id: int) -> list[tuple[int, int]]:
        """Devolve a lista de posições (row, col) pertencentes a uma região."""
        positions = list(zip(*np.where(self.board == region_id)))
        return positions
    
    def place(self, shape: list[tuple[int, int]], origin: tuple[int, int], mark) -> 'Board':
        """Retorna uma nova instância de Board com a peça colocada, marcada com `mark`."""
        new_board = np.copy(self.board)
        for dr, dc in shape:
            r, c = origin[0] + dr, origin[1] + dc
            new_board[r, c] = mark
        return Board(new_board)

    def adjacent_regions(self, region:int) -> list:
        """Devolve uma lista das regiões que fazem fronteira com a região enviada no argumento."""
        #TODO
        pass
    
    def adjacent_positions(self, row:int, col:int) -> list:
        """Devolve as posições adjacentes à região, em todas as direções, incluindo diagonais."""
        #TODO
        pass

    def adjacent_values(self, row:int, col:int) -> list:
        """Devolve os valores das celulas adjacentes à região, em todas as direções, incluindo diagonais."""
        #TODO
        pass
    
    @staticmethod
    def parse_instance() -> 'Board':
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 pipe.py < test-01.txt

            > from sys import stdin
            > line = stdin.readline().split()
        """
        board_list = []

        for line in stdin.readlines():
            line_ar = [elem for elem in line.split()]
            board_list.append(line_ar)
        board = np.array(board_list, dtype = 'U1')

        return Board(board)

    # TODO: outros metodos da classe Board

class Nuruomino(Problem):
    def __init__(self, board: Board):
        self.state = NuruominoState(board)

    def actions(self, state: NuruominoState):
        """Gera todas as formas válidas de colocar uma peça no estado atual."""
        actions = []
        board = state.board
        regions = np.unique(board.board)
        for region_id in regions:
            region_cells = board.region_cells(region_id)
            for origin in region_cells:
                for shape_group, mark in zip(
                    [L_SHAPES, I_SHAPES, T_SHAPES, S_SHAPES],
                    ['L', 'I', 'T', 'S']
                ):
                    for shape in shape_group:
                        if board.can_place(shape, origin, region_id, mark):
                            actions.append((region_id, shape, origin, mark))
        return actions

    def result(self, state: NuruominoState, action) -> NuruominoState:
        """Retorna o estado resultante de executar a 'action' sobre 'state'."""
        region_id, shape, origin, mark = action
        new_board = state.board.place(shape, origin, mark)
        return NuruominoState(new_board)
        

    def goal_test(self, state: NuruominoState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        #TODO
        pass 

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

if __name__ == "__main__":
    # Ler o tabuleiro do standard input e cria uma instância da classe Board
    problem_board = Board.parse_instance()
    
    print(problem_board)
    # Criar uma instância do problema
    problem = Nuruomino(problem_board)

    # Criar o estado inicial do problema
    initial_state = NuruominoState(problem_board)
    s1 = problem.result(initial_state, (np.str_('1'), frozenset({(1, 0), (0, 1), (2, 0), (0, 0)}), (np.int64(0), np.int64(0)), 'L'))
    s2 = problem.result(s1,(np.str_('5'), frozenset({(1, 0), (2, 0), (0, 0), (3, 0)}), (np.int64(2), np.int64(5)), 'I'))
    s3 = problem.result(s2,(np.str_('4'), frozenset({(0, 1), (1, 0), (0, 2), (0, 0)}), (np.int64(4), np.int64(0)), 'L'))
    s4 = problem.result(s3,(np.str_('2'), frozenset({(0, 1), (1, 0), (1, 1), (2, 1)}), (np.int64(0), np.int64(2)), 'T'))
    s5 = problem.result(s4, (np.str_('3'), frozenset({(1, 0), (0, 1), (2, 0), (0, 0)}), (np.int64(0), np.int64(4)), 'L'))
    print(s3.board)
    print(s4.board)
    print(s5.board)
    print(problem.actions(s5))