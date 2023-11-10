import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import trange
from collections import defaultdict
import chess
import chess.pgn
import numpy as np
import tensorflow as tf
from tqdm import trange
from tensorflow.keras import layers
import os
from tensorflow.keras.layers import Add

MODEL_NAME = "/content/drive/MyDrive/chessy/chess_model_super"
transposition_table = {}
state_cache = {}


def board_to_array(board):
    if board is None:
        return None
    piece_map = {'p': -1, 'P': 1, 'n': -2, 'N': 2, 'b': -3, 'B': 3,
                 'r': -4, 'R': 4, 'q': -5, 'Q': 5, 'k': -6, 'K': 6, '.': 0}
    board_str = str(board).replace(' ', '').replace('\n', '')
    board_array = np.array([piece_map[piece] for piece in board_str])
    board_array = board_array.reshape(8, 8, 1)
    return board_array

def create_or_load_model():
    if os.path.exists(MODEL_NAME):
        model = tf.keras.models.load_model(MODEL_NAME)
    else:
        inputs = layers.Input(shape=(8, 8, 1))

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, inputs])
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Flatten()(x)
        
        x = layers.Dense(2048, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(2048, activation='relu')(x)
        x = layers.Dropout(0.5)(x)

        outputs = layers.Dense(1, activation='tanh')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.001,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
    return model

# Modelo antigo!
# def create_or_load_model():
#     if os.path.exists(MODEL_NAME):
#         model = tf.keras.models.load_model(MODEL_NAME)
#     else:
#         inputs = layers.Input(shape=(8, 8, 1))
#         x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
#                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
#         x2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
#                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(x1)
#         x3 = Add()([x1, x2])
#         x4 = layers.MaxPooling2D((2, 2), padding='same')(x3)
#         x5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
#                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(x4)
#         x6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
#                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(x5)
#         x7 = Add()([x5, x6])
#         x8 = layers.MaxPooling2D((2, 2), padding='same')(x7)
#         x9 = layers.Flatten()(x8)
#         x10 = layers.Dense(512, activation='relu',
#                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(x9)
#         x11 = layers.Dropout(0.5)(x10)
#         x12 = layers.Dense(
#             512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x11)
#         x13 = layers.Dropout(0.5)(x12)
#         x14 = Add()([x10, x12])
#         outputs = layers.Dense(1)(x14)
#         model = tf.keras.Model(inputs=inputs, outputs=outputs)
#         initial_learning_rate = 0.001
#         lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#             initial_learning_rate,
#             decay_steps=100000,
#             decay_rate=0.96,
#             staircase=True)

#         optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
#         model.compile(optimizer=optimizer, loss='mean_squared_error')

#     return model


def augment_board(board_state):
    augmented_states = []

    augmented_states.append(board_state)

    for _ in range(3):
        board_state = np.rot90(board_state)
        augmented_states.append(board_state)

    augmented_states.append(np.flipud(board_state))

    augmented_states.append(np.fliplr(board_state))

    return augmented_states


PIECE_VALUES = {
    'P': 1,
    'N': 3,
    'B': 3,
    'R': 5,
    'Q': 9,
    'K': 50, 
}


def move_ordering_heuristic(move, board):
    score = 0

    if board.is_capture(move):
        moving_piece = board.piece_at(move.from_square).symbol().upper()
        captured_piece = board.piece_at(move.to_square)

        if captured_piece is not None:
            captured_piece = captured_piece.symbol().upper()
            score += 10 * \
                (PIECE_VALUES[captured_piece] - PIECE_VALUES[moving_piece])
        else:
            score += 10

    if move.promotion:
        score += 90

    board.push(move)
    if board.is_check():
        score += 30
    if board.is_checkmate():
        score += 900
    board.pop()

    score += len(list(board.legal_moves)) * 0.1

    if move.to_square in [27, 28, 35, 36]:
        score += 5

    return score

def minimax(board, depth, maximizing, alpha, beta, model):
    board_fen = board.fen() + (" w" if board.turn else " b")

    if board_fen in transposition_table:
        stored_value, stored_alpha, stored_beta, stored_move = transposition_table[board_fen]

        if stored_alpha <= alpha and stored_beta >= beta:
            return stored_value, stored_move

    # Base case
    if depth == 0 or board.is_game_over():
        if board_fen not in state_cache:
            board_state = board_to_array(board).reshape(1, 8, 8, 1)
            value = model.predict(board_state, verbose=0)[0][0]
            state_cache[board_fen] = value
        return state_cache[board_fen], None

    moves = list(board.legal_moves)

    moves.sort(key=lambda move: move_ordering_heuristic(
        move, board), reverse=maximizing)

    best_move = None
    best_eval = float('-inf') if maximizing else float('inf')

    batch_states = []
    for move in moves:
        future_board = board.copy()
        future_board.push(move)
        future_board_fen = future_board.fen() + (" w" if future_board.turn else " b")

        if future_board_fen not in state_cache:
            board_state = board_to_array(future_board).reshape(1, 8, 8, 1)
            batch_states.append((future_board_fen, board_state))

    if batch_states:
        batch_input = np.vstack([state for _, state in batch_states])
        batch_output = model.predict(batch_input, verbose=0)

        for i, (future_board_fen, _) in enumerate(batch_states):
            state_cache[future_board_fen] = batch_output[i][0]

    for move in moves:
        future_board = board.copy()
        future_board.push(move)
        future_board_fen = future_board.fen() + (" w" if future_board.turn else " b")  # Add turn indicator

        eval = state_cache.get(future_board_fen, None)
        if eval is None:
            eval, _ = minimax(future_board, depth - 1,
                              not maximizing, alpha, beta, model)

        if maximizing:
            if eval > best_eval:
                best_eval = eval
                best_move = move
            alpha = max(alpha, eval)
        else:
            if eval < best_eval:
                best_eval = eval
                best_move = move
            beta = min(beta, eval)

        if beta <= alpha:
            break

    transposition_table[board_fen] = (best_eval, alpha, beta, best_move)
    return best_eval, best_move


def custom_heuristic(board):
    score = 0

    piece_values = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
                    'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000}

    pawn_table = [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, -20, -20, 10, 10, 5,
        5, -5, -10, 0, 0, -10, -5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, 5, 10, 25, 25, 10, 5, 5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
        0, 0, 0, 0, 0, 0, 0, 0
    ]

    knight_table = [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50
    ]

    bishop_table = [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    ]

    rook_table = [
        0, 0, 0, 5, 5, 0, 0, 0,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        5, 10, 10, 10, 10, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0
    ]

    queen_table = [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20
    ]

    king_table = [
        20, 30, 10, 0, 0, 10, 30, 20,
        20, 20, 0, 0, 0, 0, 20, 20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30
    ]

    isolated_pawns = [False] * 8
    passed_pawns = [False] * 8

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_score = piece_values.get(piece.symbol(), 0)
            score += piece_score
            file_index = chess.square_file(square)

            if piece.piece_type == chess.PAWN:
                isolated_pawns[file_index] = True
                if all(board.pieces(chess.PAWN, not piece.color) & chess.BB_FILES[f] == chess.BB_EMPTY for f in [file_index - 1, file_index + 1] if 0 <= f <= 7):
                    passed_pawns[file_index] = True

                score += pawn_table[square] if piece.color == chess.WHITE else - \
                    pawn_table[chess.square_mirror(square)]

            elif piece.piece_type == chess.KNIGHT:
                score += knight_table[square] if piece.color == chess.WHITE else - \
                    knight_table[chess.square_mirror(square)]

            elif piece.piece_type == chess.BISHOP:
                score += bishop_table[square] if piece.color == chess.WHITE else - \
                    bishop_table[chess.square_mirror(square)]

            elif piece.piece_type == chess.ROOK:
                score += rook_table[square] if piece.color == chess.WHITE else - \
                    rook_table[chess.square_mirror(square)]

            elif piece.piece_type == chess.QUEEN:
                score += queen_table[square] if piece.color == chess.WHITE else - \
                    queen_table[chess.square_mirror(square)]

            elif piece.piece_type == chess.KING:
                score += king_table[square] if piece.color == chess.WHITE else - \
                    king_table[chess.square_mirror(square)]

    if len(board.pieces(chess.BISHOP, chess.WHITE)) == 2:
        score += 50
    if len(board.pieces(chess.BISHOP, chess.BLACK)) == 2:
        score -= 50

    king_safety = 0
    king_square = board.king(chess.WHITE)
    if king_square is not None:
        adjacent_squares = [king_square + i for i in [-9, -
                                                      8, -7, -1, 1, 7, 8, 9] if 0 <= king_square + i < 64]
        for square in adjacent_squares:
            if board.is_attacked_by(chess.BLACK, square):
                king_safety -= 50
    score += king_safety

    mobility = sum(1 for _ in board.legal_moves)
    score += 0.1 * mobility

    if chess.BB_RANK_7 & board.pieces(chess.ROOK, chess.WHITE).mask:
        score += 20
    if chess.BB_RANK_2 & board.pieces(chess.ROOK, chess.BLACK).mask:
        score -= 20

    for i, isolated in enumerate(isolated_pawns):
        if isolated:
            if i > 0 and isolated_pawns[i - 1]:
                continue
            if i < 7 and isolated_pawns[i + 1]:
                continue
            pawns_on_file = board.pieces(
                chess.PAWN, chess.WHITE) & chess.BB_FILES[i]
            score -= 20 * bin(pawns_on_file).count("1")
            pawns_on_file = board.pieces(
                chess.PAWN, chess.BLACK) & chess.BB_FILES[i]
            score += 20 * bin(pawns_on_file).count("1")

    for i, passed in enumerate(passed_pawns):
        if passed:
            pawns_on_file = board.pieces(
                chess.PAWN, chess.WHITE) & chess.BB_FILES[i]
            score += 30 * bin(pawns_on_file).count("1")
            pawns_on_file = board.pieces(
                chess.PAWN, chess.BLACK) & chess.BB_FILES[i]
            score -= 30 * bin(pawns_on_file).count("1")

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.ROOK:
            file_index = chess.square_file(square)
            pawns_on_file = board.pieces(
                chess.PAWN, piece.color) & chess.BB_FILES[file_index]
            enemy_pawns_on_file = board.pieces(
                chess.PAWN, not piece.color) & chess.BB_FILES[file_index]
            if not pawns_on_file:
                score += 25 if not enemy_pawns_on_file else 12.5  # Open and semi-open file bonus
    for move in board.legal_moves:
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                score += piece_values.get(captured_piece.symbol(), 0)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.WHITE:
                score += 0.1  
            else:
                score -= 0.1  
    return score

def sort_moves(board, moves):
    piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
                    'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}  

    def move_value(move):
        target_piece = board.piece_at(move.to_square)
        if target_piece:
            return piece_values.get(target_piece.symbol(), 0)
        else:
            return 0

    sorted_moves = sorted(moves, key=move_value, reverse=True)
    return sorted_moves


transposition_table_2 = {}
def minimax_heuristic(board, depth, maximizing, alpha, beta):
    board_fen = board.fen() + (" w" if board.turn else " b")  

    if board_fen in transposition_table_2:
        return transposition_table_2[board_fen]

    if depth == 0 or board.is_game_over():
        return custom_heuristic(board), None

    moves = list(board.legal_moves)
    moves = sort_moves(board, moves)

    best_move = None
    best_eval = float('-inf') if maximizing else float('inf')

    for move in moves:
        future_board = board.copy()
        future_board.push(move)
        eval, _ = minimax_heuristic(
            future_board, depth - 1, not maximizing, alpha, beta)

        if maximizing:
            if eval > best_eval:
                best_eval = eval
                best_move = move
            alpha = max(alpha, eval)
        else:
            if eval < best_eval:
                best_eval = eval
                best_move = move
            beta = min(beta, eval)

        if beta <= alpha:
            break

    transposition_table_2[board_fen] = (best_eval, best_move)
    return best_eval, best_move

def custom_reward(board, state_count, draw_penalty=-5):
    reward = 0
    PIECE_VALUES = {'p': -1, 'P': 1, 'n': -3, 'N': 3, 'b': -3, 'B': 3, 'r': -5, 'R': 5, 'q': -9, 'Q': 9, 'k': -20, 'K': 20}
    
    KING_INITIAL_POSITIONS = {chess.E1, chess.E8}
    
    if board.king(chess.WHITE) not in KING_INITIAL_POSITIONS:
        reward -= 1  
    if board.king(chess.BLACK) not in KING_INITIAL_POSITIONS:
        reward += 1  
    
    total_value = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            total_value += PIECE_VALUES.get(piece.symbol(), 0)
    
    reward += total_value // 10

    if board.is_capture(board.peek()):
        if total_value < 0 and board.turn:
            reward -= 2
        elif total_value > 0 and not board.turn:
            reward += 2
    
    if board.is_check():
        reward -= 3 if board.turn else 3  
    
    if board.is_capture(board.peek()):
        reward += 1 if board.turn else -1  
    
    if state_count[board.fen()] > 1:
        reward -= 10  

    if board.can_claim_threefold_repetition():
        reward += draw_penalty

    return reward


def self_play_and_train_batched_with_minimax(model, num_games=100, num_accumulated_games=32, depth=20, epsilon=0.0):
    accumulated_states = []
    accumulated_evals = []
    state_count = defaultdict(int)

    outer_pbar = trange(num_games, desc="Overall Progress", leave=False)

    for i in outer_pbar:
        board = chess.Board()
        game_states = []
        game_evals = []

        while not board.is_game_over():
            state_count[board.fen()] += 1
            maximizing = board.turn
            if board.turn:
                best_value, best_move = minimax(board, depth, maximizing, float('-inf'), float('inf'), model)
            else:
                best_value, best_move = minimax_heuristic(board, depth, maximizing, float('-inf'), float('inf'))

            board_array = board_to_array(board)
            augmented_states = augment_board(board_array)
            game_states.extend(augmented_states)

            board.push(best_move)

            reward = custom_reward(board, state_count)
            game_evals.extend([reward] * len(augmented_states))

        if board.is_game_over():
            game_outcome, final_reward = determine_game_outcome_and_reward(board)
            game_evals = [intermediate + final_reward for intermediate in game_evals]

        accumulated_states.extend(game_states)
        accumulated_evals.extend(game_evals)

        if (i + 1) % num_accumulated_games == 0:
            X = np.array(accumulated_states)
            y = np.array(accumulated_evals)
            history = model.fit(X, y, epochs=32, verbose=0, batch_size=min(32, len(X)))
            model.save(MODEL_NAME)
            outer_pbar.set_postfix({"Game": i + 1, "Loss": history.history['loss'][-1], "Outcome": game_outcome})

            accumulated_states.clear()
            accumulated_evals.clear()
            state_count.clear()
            transposition_table.clear()
            state_cache.clear()

def determine_game_outcome_and_reward(board):
    if board.is_checkmate():
        return "Checkmate", 50 if board.turn else -50
    elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
        return "Draw", -10
    elif board.is_variant_draw():
        return "Draw", -10
    else:
        return "Unknown", -10


def play_1v1_with_visualization_heuristic(model, depth=4, first=False):
    board = chess.Board()
    game = chess.pgn.Game()
    game.setup(board.fen())
    node = game

    while not board.is_game_over():
        maximizing = board.turn
        best_value, best_move = None, None

        if first:
            if maximizing:
                best_value, best_move = minimax_heuristic(
                    board, depth, maximizing, float('-inf'), float('inf'))
            else:  
                best_value, best_move = minimax(
                    board, depth, maximizing, float('-inf'), float('inf'), model)
        else:
            if not maximizing:  
                best_value, best_move = minimax_heuristic(
                    board, depth, maximizing, float('-inf'), float('inf'))
            else: 
                best_value, best_move = minimax(
                    board, depth, maximizing, float('-inf'), float('inf'), model)

        print(f"Move: {best_move}, Value: {best_value}")

        board.push(best_move)
        new_node = node.add_variation(best_move)
        node = new_node

    print("Game Over. Reason:", board.result())
    print("Full PGN of the game:")
    exporter = chess.pgn.StringExporter(headers=True, variations=True)
    pgn_string = str(game.accept(exporter))
    print(pgn_string)


if __name__ == "__main__":
    model = create_or_load_model()
    self_play_and_train_batched_with_minimax(model, num_games=1000, depth=5)
    print("Starting 1v1 game with visualization...")
    play_1v1_with_visualization_heuristic(model, depth=2, first=False)
