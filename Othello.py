import numpy as np
import bisect


class Board(object):
    def __init__(self):
        """
            Constructor of Board object.
        """
        self.board: str = "___________________________WB______BW___________________________"
        self.tiles: int = 4
        self.blackTiles: int = 2
        self.full: bool = False

    def placeTile(self, player: str, position: tuple):
        '''
        Places the tile corresponding to player in position given by tuple.
        :param player: player_sign
        :param position: tuple (i, j)
        :return: void
        '''
        assert player in ("B", "W"), "ERROR: Player not defined"
        assert position[0] >= 0 and position[0] < 8 and position[1] >= 0 and position[1] < 8, "ERROR: Invalid position"

        index = position[0] * 8 + position[1]
        if self.board[index] != player:
            self.board = self.board[:index] + player + self.board[index + 1:]
            self.tiles += 1
            if player == "B": self.blackTiles += 1

        if self.tiles == 64: self.full = True

    def makeMovement(self, player: str, movement: tuple):
        '''
        Applies the movement given by movement to the board.
        :param player: str, 'W' or 'B'
        :param movement: tuple with the format (init, (length, direction), target)
        '''
        # MOVE = (origin, (n_tiles, direction), end)
        self.recMovement(player, movement[0][0], movement[0][1], movement[1][1], movement[2])

    def recMovement(self, player: str, i: int, j: int, direction: int, final):
        '''
        Applies the movement recursively.
        :param player: 'W' or 'B'
        :param i: current row
        :param j: current column
        :param direction: int from 0 to 7
        :param final: target cell of the movement
        '''
        if i == final[0] and j == final[1]:
            self.placeTile(player, (i, j))
            return 0
        else:
            if self[i][j] != player:
                index = i * 8 + j
                self.board = self.board[:index] + player + self.board[index + 1:]
                if player == 'B':
                    self.blackTiles += 1
                else:
                    self.blackTiles -= 1
            if direction == 0: return self.recMovement(player, i - 1, j, direction, final)
            if direction == 1: return self.recMovement(player, i - 1, j + 1, direction, final)
            if direction == 2: return self.recMovement(player, i, j + 1, direction, final)
            if direction == 3: return self.recMovement(player, i + 1, j + 1, direction, final)
            if direction == 4: return self.recMovement(player, i + 1, j, direction, final)
            if direction == 5: return self.recMovement(player, i + 1, j - 1, direction, final)
            if direction == 6: return self.recMovement(player, i, j - 1, direction, final)
            if direction == 7: return self.recMovement(player, i - 1, j - 1, direction, final)

    def __getitem__(self, item):
        '''
        OVerriding method for getting the itemth elemtn of the board
        :param item: int from 0 to 7
        :return: row given by item
        '''
        return self.board[8 * item:8 * item + 8]

    def __str__(self):
        '''
        Overriding method for printing Board Object
        :return: str representing the state of the board
        '''
        tiles = "Black Tiles: " + str(self.blackTiles) + " - White Tiles: " + str(self.tiles - self.blackTiles) + "\n\n"
        board = "  A B C D E F G H\n"
        for i in range(64):
            if i % 8 == 0: board += str(i // 8 + 1) + " "
            board += self.board[i]
            if i % 8 == 7:
                board += " " + str(i // 8 + 1) + "\n"
            else:
                board += ' '
        board += "  A B C D E F G H\n"
        return tiles + board

    def printWithMovements(self, movements):
        '''
        Prints the board signaling the target cells of the movements
        :param movements: list of movements
        :return: str
        '''
        moves = [move[0][2][0]*8+move[0][2][1] for move in movements]
        tiles = "Black Tiles: " + str(self.blackTiles) + " - White Tiles: " + str(self.tiles - self.blackTiles) + "\n\n"
        board = "  A B C D E F G H\n"
        for i in range(64):
            if i % 8 == 0: board += str(i // 8 + 1) + " "
            if i in moves:
                board += '*'
            else: board += self.board[i]
            if i % 8 == 7:
                board += " " + str(i // 8 + 1) + "\n"
            else:
                board += ' '
        board += "  A B C D E F G H\n"
        return tiles + board

    def __copy__(self):
        '''
        Create a new Board object identicall to the caller
        :return: Board
        '''
        newBoard = Board()
        newBoard.full = self.full
        newBoard.board = self.board
        newBoard.tiles = self.tiles
        newBoard.blackTiles = self.blackTiles
        return newBoard

    def loadValues(self, values: tuple):
        '''
        Loads the values passed as tuple into the board.
        :param values:
        '''
        self.board = values[0]
        self.tiles = values[1]
        self.blackTiles = values[2]
        self.full = values[3]

def generateMoves(player, board: Board):
    '''
    Returns a list of allowed movements for given player
    :param player: player_sign
    :param board: Board
    :return: list
    '''
    assert player in ("B", "W"), "ERROR: Player not defined"
    # MOVE = (origin, (n_tiles, direction), end)
    moves = []
    for i in range(64):
        if board.board[i] == '_':
            mov = []
            for direction in range(8):
                move = recMove(board, direction, player, i // 8, i % 8, 0)
                if move != None: mov.append((move[0], move[1], (i // 8, i % 8)))
            if mov != []: moves.append(mov)
    return moves

def recMove(board: Board, direction: int, player, i, j, offset):
    '''
    Recursively finds the movements that a player can perfomr.
    :param board: Board
    :param direction: int from 0 to 7
    :param player: 'W' or 'B'
    :param i: current row
    :param j: current column
    :param offset: movement index
    :return: tuple
    '''
    assert direction in (0, 1, 2, 3, 4, 5, 6, 7), "ERROR: wrong direction"
    if i < 0 or i > 7 or j < 0 or j > 7: return None
    if offset == 1 and (board[i][j] == player or board[i][j] == '_'): return None
    if board[i][j] == player and offset > 1: return ((i, j), (offset - 1, (direction + 4) % 8))

    if board[i][j] != '_' or offset == 0:
        if direction == 0: return recMove(board, direction, player, i - 1, j, offset + 1)
        if direction == 1: return recMove(board, direction, player, i - 1, j + 1, offset + 1)
        if direction == 2: return recMove(board, direction, player, i, j + 1, offset + 1)
        if direction == 3: return recMove(board, direction, player, i + 1, j + 1, offset + 1)
        if direction == 4: return recMove(board, direction, player, i + 1, j, offset + 1)
        if direction == 5: return recMove(board, direction, player, i + 1, j - 1, offset + 1)
        if direction == 6: return recMove(board, direction, player, i, j - 1, offset + 1)
        if direction == 7: return recMove(board, direction, player, i - 1, j - 1, offset + 1)

class Game(object):
    def __init__(self, player1, player2):
        '''
        Constructor of Game object.
        :param player1: Player
        :param player2: Player
        '''
        self.board = Board()
        self.players = [player1, player2]

    def start(self):
        '''
        Initiates the game.
        '''
        i = 1
        while not self.board.full:
            if not self.players[0].canPlay and not self.players[1].canPlay: break
            currentPlayer = self.players[(i-1) % 2]
            acceptedMoves = generateMoves(currentPlayer.player, self.board)
            print("ROUND " + str(i) + " ------ " + currentPlayer.player + " Player's Turn")
            print(self.board.printWithMovements(acceptedMoves))

            if len(acceptedMoves) == 0:
                print("You can't make any movements.")
                i += 1
                currentPlayer.canPlay = False
                continue
            else:
                currentPlayer.canPlay = True

            chosenMove = ()

            if isinstance(currentPlayer, mePlayer):
                chosenMove = currentPlayer.play()
            if isinstance(currentPlayer, CPU):
                chosenMove = currentPlayer.play(self.board, i);

            if chosenMove == (None): continue
            valid = False
            for movements in acceptedMoves:
                # MOVEMENT = (start, (no_tiles, direction), end)
                for movement in movements:
                    if movement[2][0] == chosenMove[0] and movement[2][1] == chosenMove[1]:
                        valid = True
                        self.board.makeMovement(currentPlayer.player, movement)
                    else:
                        continue
            if valid:
                i = (i + 1)

        print("GAME OVER.")
        print(self.board)

class CPUConfig(object):
    def __init__(self, param: dict):
        '''
        Constructor of CPUConfig object from the configuration parameters given in param
        :param param: dict with configuration of depth, alphabeta, evFunction
        '''
        self.depth = param['depth']
        self.alphabeta = param['alphabeta']
        self.evFunction = param['evFunction']
        self.bonusMap = [[99, -8, 8, 6],
                        [-8, -24, -4, -3],
                        [8, -4, 7, 4],
                        [6, -3, 4, 0]]
        # (player_mobility_advantage, player_tiles_advantage, player_positional_strategy_advantage)
        self.coefficients = (0, 0, 0)

    def evaluate(self, player, position: Board):
        '''
        Returns the evaluation of the board with respect to a player according to the evaluation parameter.
        :param player: 'W' or 'B'
        :param position: Board
        :return: float evaluation
        '''
        if self.evFunction == "ABSOLUTE": return self.absoluteEvaluation(player, position)
        if self.evFunction == "RELATIVE": return self.relativeEvaluation(player, position)

    def getDepth(self, round: int):
        '''
        Calculates the depth of search tree at a given round.
        :param round: int
        :return: int
        '''
        if self.depth != "VARIABLE": return self.depth
        minimun_depth = 3
        if round <= 5: return 5
        if round > 5 and round <= 10: return 4
        if round > 10 and round <= 50: return 3
        if round > 40: return 4

    def absoluteEvaluation(self, player, position: Board):
        '''
        Absolute evaluation of the board with respect to player.
        :param player: 'W' or 'B'
        :param position: Board
        :return: float, absolut evaluation of board.
        '''

        # Parameters of the evaluation
        bonus_for_position = 0

        if player == "B":
            disk_Count = position.blackTiles
        else:
            disk_Count = position.tiles - position.blackTiles

        legal_Moves = len(generateMoves(player, position))

        for i in range(8):
            for j in range(8):
                if position[i][j] == player:
                    bonus_for_position += self.getCellBonus(i, j)
        return np.dot(self.coefficients, (legal_Moves, disk_Count, bonus_for_position))

    def relativeEvaluation(self, player, position: Board):
        '''
        Returns board evaluation as a relation between the two players from the perspective of player.
        :param player: 'W' or 'B'
        :param position: BOard
        :return: float, relative evaluation of board.
        '''
        assert player in ("W", "B"), "ERROR: invalid player."
        #MOBILITY; NUMBER_OF_DISKS; POSITIONAL_STRATEGY
        opponent = [op for op in ("W", "B") if op != player][0]

        player_mobility = len(generateMoves(player, position))
        opponent_mobility = len(generateMoves(opponent, position))

        if player == "B":
            player_tiles = position.blackTiles
            opponent_tiles = position.tiles - player_tiles
        else:
            opponent_tiles = position.blackTiles
            player_tiles = position.tiles - opponent_tiles

        player_positional_strategy = 0
        opponent_positional_strategy = 0

        for i in range(8):
            for j in range(8):
                if position[i][j] == player: player_positional_strategy += self.getCellBonus(i, j)
                if position[i][j] == opponent: opponent_positional_strategy += self.getCellBonus(i, j)
        try:
            player_mobility_advantage = player_mobility/(player_mobility+opponent_mobility)
        except:
            player_mobility_advantage = 0
        player_tiles_advantage = player_tiles/(player_tiles+opponent_tiles)
        try:
            player_positional_strategy_advantage = player_positional_strategy/(player_positional_strategy+opponent_positional_strategy)
        except: player_positional_strategy_advantage = 0

        return np.dot(self.coefficients, (player_mobility_advantage, player_tiles_advantage, player_positional_strategy_advantage))

    def getCellBonus(self, i, j):
        '''
        Calculates the bonus associated with cell (i,j), applying symmetry of the board to self.bonusMap.
        :param i: int, cell row
        :param j: int, cell column
        :return: bonus of cell
        '''

        new_i: int
        new_j: int

        if i > 3: new_i = (3 - i) % 4
        else: new_i = i
        if j > 3: new_j = (3 - j) % 4
        else: new_j = j

        return self.bonusMap[new_i][new_j]

    def getProbabilityThereIsACellWithHigherBonusThan(self, x, round):
        '''
        Returns the probability that there is a achievable cell with higher bonus than x at round.
        :param x: float
        :param round: bonus
        :return: float
        '''
        numberOfCells = 0
        for i in self.bonusMap:
            for j in i:
                if j > x: numberOfCells += 4
        return numberOfCells/(64*(0.6*round))

    def generateCoefficients(self, round: int):
        '''
        Yields the coefficients applied at round.
        :param round: int
        '''
        #(player_mobility_advantage, player_tiles_advantage, player_positional_strategy_advantage))
        if round < 30:
            self.coefficients = (50, -10, 5)
        elif round < 40:
            self.coefficients = (10, 10, 5)
        else:
            self.coefficients = (5, 30, 3)


class Player(object):
    def __init__(self, sign):
        '''
        Constructor of Player Object
        :param sign: 'W' or 'B'
        '''
        assert sign in ("W", "B"), "ERROR: Player not recognized"
        self.player = sign
        self.opponent: str
        if sign == "W": self.opponent = "B"
        else: self.opponent = "W"
        self.canPlay = True

class CPU(Player):
    def setConfig(self, config: CPUConfig):
        '''
        Loads configuration in config of player.
        :param config: CPUConfig
        '''
        self.config = config

    def play(self, board: Board, round):
        '''
        Runs minimax algorithm with the configuration specified in config file.
        :param board: Board
        :return: string move done by the CPU Player
        '''
        depth = self.config.getDepth(round)
        alphabeta = self.config.alphabeta
        self.round = round
        self.config.generateCoefficients(round)
        if alphabeta:
            move = self.minimax(board, depth, True, 0, float('-inf'), float('inf'))[0][2]
        else:
            move = self.minimax(board, depth, True, 0, None, None)[0][2]
        print(move)
        return move

    def minimax(self, position: Board, depth, maxPlayer, i, alpha=None, beta=None):
        '''
            Performs minimax algorithm.
            Alpha Beta pruning is performed if alpha & beta != None.
            :param position: Board
            :param depth: int
            :param maxPlayer: boolean
            :param alpha: int
            :param beta: int
            '''
        if maxPlayer: moves = generateMoves(self.player, position)
        else: moves = generateMoves(self.opponent, position)


        if depth == 0 or position.tiles == 64 or len(moves) == 0:
            return self.config.evaluate(self.player, position)

        board_Copy = position.__copy__()
        values_Copy = (position.board, position.tiles, position.blackTiles, position.full)

        if i == 0: bestMove = ()
        if maxPlayer:
            maxEval = float("-inf")
            for movements in self.yieldMovements(moves, self.round):
                for movement in movements:
                    board_Copy.makeMovement(self.player, movement)
                eval = self.minimax(board_Copy, depth - 1, False, i + 1, alpha, beta)
                if i == 0:
                    if max(maxEval, eval) != maxEval: bestMove = movements[:]
                maxEval = max(maxEval, eval)
                if alpha != None and beta != None:
                    alpha = max(alpha, eval)
                    if beta <= alpha: break

                board_Copy.loadValues(values_Copy)
            if i == 0: return bestMove
            return maxEval

        else:
            minEval = float("inf")
            for movements in self.yieldMovements(moves, self.round):
                for movement in movements:
                    board_Copy.makeMovement(self.opponent, movement)
                eval = self.minimax(board_Copy, depth - 1, True, i + 1, alpha, beta)
                if i == 0:
                    if min(minEval, eval) != minEval: bestMove = movements[:]
                minEval = min(minEval, eval)
                if alpha != None and beta != None:
                    beta = min(beta, eval)
                    if beta <= alpha: break

                board_Copy.loadValues(values_Copy)
            if i == 0: return bestMove
            return minEval

    def yieldMovements(self, possible_moves, round: int):
        '''
        Receives a list of moves, and yields them in the order specified by a heuristic.
        :param possible_moves: list of movements
        :param round:  int
        :return: movement
        '''
        moves = []
        keys = []

        total_moves = len(possible_moves)
        seen_moves = 0

        for movement in possible_moves:
            #Assumes every movement is a 1..* list of movements to the same cell
            seen_moves += 1
            target: tuple = movement[0][2] #Target cell of the movement
            cell_bonus = self.config.getCellBonus(target[0], target[1])
            p = self.config.getProbabilityThereIsACellWithHigherBonusThan(cell_bonus, round)*(total_moves-seen_moves)/total_moves
            if p > 0.3:
                eval = cell_bonus+sum([i[1][0] for i in movement])
                i = bisect.bisect_left(keys, eval)
                keys.insert(i, eval)
                moves.insert(i, movement)
            else:
                yield movement
        while len(moves) > 0:
            yield moves.pop(len(moves)-1)

    def yieldMovements2(self, possible_moves, round: int):
        '''
        Yields the movements as provided in the array.
        :param possible_moves: list of movements
        :param round: int
        :return: movement tuple
        '''
        for i in possible_moves:
            yield i

class mePlayer(Player):
    def play(self):
        '''
        Asks the user to play.
        :return: Move entered by user
        '''
        chosenMove = input("What's your next movement?")  # Universal format
        chosenMove = self.move(chosenMove.upper())
        return chosenMove

    def move(self, movement: str) -> tuple:
        '''
        Converts movement inut by the user into usable format (row, column)
        :param movement: str in the format "letter/number" or "number/letter"
        :return: tuple with move.
        '''
        assert len(movement) == 2, "ERROR: invalid movement"
        letters = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
        if movement[0] in letters:
            # FORMAT = letter/number A5 -> (4, 0)
            return (int(movement[1]) - 1, letters[movement[0]])
        if movement[1] in letters:
            # FORMAT = number/letter 5A -> (4, 0)
            return (int(movement[0]) - 1, letters[movement[1]])
        print("Invalid Movement")
        return (None)

#Create a configure a CPUPLayer
player1Config = CPUConfig({'depth': 3, 'alphabeta':True, 'evFunction':'ABSOLUTE'})
cpuplayer1 = CPU('B')
cpuplayer1.setConfig(player1Config)

#Create a configure a CPUPLayer
player2Config = CPUConfig({'depth': 3, 'alphabeta':True, 'evFunction':'ABSOLUTE'})
cpuplayer2 = CPU('W')
cpuplayer2.setConfig(player2Config)

#Create a user-played player.
me = mePlayer("W")

#Create game
game = Game(cpuplayer1, me)
print(game.board)
game.start()