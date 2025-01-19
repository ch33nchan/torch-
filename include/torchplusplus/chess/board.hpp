#pragma once
#include <array>
#include <string>
#include <vector>

namespace torchplusplus {
namespace chess {

enum class Piece {
    EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
};

enum class Color {
    WHITE, BLACK
};

struct Move {
    int from_square;
    int to_square;
    Piece promotion;
    
    Move(int from, int to, Piece prom = Piece::EMPTY) 
        : from_square(from), to_square(to), promotion(prom) {}
};

class Board {
public:
    Board();
    
    // Board manipulation
    void make_move(const Move& move);
    void undo_move();
    std::vector<Move> generate_legal_moves() const;
    bool is_game_over() const;
    
    // Position evaluation
    float evaluate() const;
    
    // Board representation
    std::string to_fen() const;
    void from_fen(const std::string& fen);
    
private:
    std::array<Piece, 64> pieces_;
    std::array<Color, 64> colors_;
    Color side_to_move_;
    bool castle_rights_[4];  // White kingside, White queenside, Black kingside, Black queenside
    int en_passant_square_;
    int halfmove_clock_;
    int fullmove_number_;
};

} // namespace chess
} // namespace torchplusplus