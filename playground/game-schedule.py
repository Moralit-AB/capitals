from mip import Model, xsum, minimize, BINARY, INTEGER, OptimizationStatus
from itertools import combinations
import numpy as np
import math

def analyze_feasibility(n_players, n_rounds=5):
    """
    Analyze theoretical bounds for 2-player game distribution.
    
    Returns:
        dict with feasibility analysis
    """
    remainder = n_players % 3
    
    if remainder == 0:
        return {
            "needs_2p_games": False,
            "games_per_round": n_players // 3,
            "two_player_games_per_round": 0,
            "total_2p_slots": 0,
            "min_max_2p_per_player": 0,
            "perfect_balance": True
        }
    
    # Remainder is 1 or 2
    # Option 1: use (n//3 - 1) 3P games and make 2P games from remainder
    if remainder == 1:
        # n = 3k + 1: use (k-1) 3P games + 2 2P games per round
        three_player_games = n_players // 3 - 1
        two_player_games = 2
    else:  # remainder == 2
        # n = 3k + 2: use k 3P games + 1 2P game per round
        three_player_games = n_players // 3
        two_player_games = 1
    
    total_2p_slots = n_rounds * two_player_games * 2
    min_max_2p = math.ceil(total_2p_slots / n_players)
    perfect_balance = (total_2p_slots % n_players == 0)
    
    return {
        "needs_2p_games": True,
        "games_per_round": three_player_games + two_player_games,
        "three_player_games_per_round": three_player_games,
        "two_player_games_per_round": two_player_games,
        "total_2p_slots": total_2p_slots,
        "min_max_2p_per_player": min_max_2p,
        "perfect_balance": perfect_balance
    }


def create_game_schedule(n_players, n_rounds=5, max_games_per_round=10):
    """
    Create optimal game schedule using ILP.
    
    Args:
        n_players: Number of players
        n_rounds: Number of rounds (default 5)
        max_games_per_round: Upper bound on games per round
    
    Returns:
        schedule: List of rounds, each round is list of games (lists of player indices)
        stats: Dictionary with solution statistics
    """
    
    feasibility = analyze_feasibility(n_players, n_rounds)
    print(f"Feasibility analysis for {n_players} players:")
    print(f"  2P games needed: {feasibility['needs_2p_games']}")
    if feasibility['needs_2p_games']:
        print(f"  Games per round: {feasibility['three_player_games_per_round']}×3P + {feasibility['two_player_games_per_round']}×2P")
        print(f"  Total 2P slots: {feasibility['total_2p_slots']}")
        print(f"  Theoretical min max_2p: {feasibility['min_max_2p_per_player']}")
        print(f"  Perfect balance possible: {feasibility['perfect_balance']}")
    print()
    
    model = Model("game_schedule")
    model.verbose = 0
    
    players = range(n_players)
    rounds = range(n_rounds)
    games = range(max_games_per_round)
    
    # Decision variables
    x = {(p, g, r): model.add_var(var_type=BINARY, name=f"x_{p}_{g}_{r}")
         for p in players for g in games for r in rounds}
    
    y = {(g, r): model.add_var(var_type=BINARY, name=f"y_{g}_{r}")
         for g in games for r in rounds}
    
    game_used = {(g, r): model.add_var(var_type=BINARY, name=f"used_{g}_{r}")
                 for g in games for r in rounds}
    
    two_player_count = {p: model.add_var(var_type=INTEGER, name=f"two_count_{p}")
                        for p in players}
    
    max_two_player = model.add_var(var_type=INTEGER, name="max_two_player")
    
    meet_count = {(p1, p2): model.add_var(var_type=INTEGER, name=f"meet_{p1}_{p2}")
                  for p1, p2 in combinations(players, 2)}
    
    max_meet = model.add_var(var_type=INTEGER, name="max_meet")
    
    # Constraints
    
    # 1. Each player in exactly one game per round
    for r in rounds:
        for p in players:
            model += xsum(x[p, g, r] for g in games) == 1
    
    # 2. Game size is 2 or 3 players (only if game is used)
    for g in games:
        for r in rounds:
            game_size = xsum(x[p, g, r] for p in players)
            model += game_size >= 2 * game_used[g, r]
            model += game_size <= 3 * game_used[g, r]
    
    # 3. Link y to game size: y[g,r]=1 iff game g in round r has 2 players
    for g in games:
        for r in rounds:
            game_size = xsum(x[p, g, r] for p in players)
            model += game_size >= 2 * y[g, r]
            model += game_size <= 2 + (1 - y[g, r]) * 1
    
    # 4. Count 2-player games per player
    for p in players:
        model += two_player_count[p] == xsum(x[p, g, r] * y[g, r] 
                                              for g in games for r in rounds)
    
    # 5. Bound max_two_player
    for p in players:
        model += max_two_player >= two_player_count[p]
    
    # 6. Count pairwise meetings
    for r in rounds:
        for g in games:
            for p1, p2 in combinations(players, 2):
                model += meet_count[p1, p2] >= x[p1, g, r] + x[p2, g, r] - 1
    
    # 7. Bound max_meet
    for p1, p2 in combinations(players, 2):
        model += max_meet >= meet_count[p1, p2]
    
    # 8. Game ordering (symmetry breaking): use lower-indexed games first
    for g in range(len(games) - 1):
        for r in rounds:
            model += game_used[g, r] >= game_used[g + 1, r]
    
    # 9. Constrain number of 2-player games per round based on player count
    if feasibility['needs_2p_games']:
        for r in rounds:
            model += xsum(y[g, r] for g in games) == feasibility['two_player_games_per_round']
    
    # Objective: Lexicographic - prioritize 2-player fairness, then meetings
    model.objective = minimize(1000 * max_two_player + max_meet)
    
    # Solve
    status = model.optimize(max_seconds=300)
    
    if status not in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:
        return None, {"status": "infeasible", "feasibility": feasibility}
    
    # Extract solution
    schedule = []
    for r in rounds:
        round_games = []
        for g in games:
            game_players = [p for p in players if x[p, g, r].x >= 0.99]
            if len(game_players) >= 2:
                round_games.append(sorted(game_players))
        schedule.append(round_games)
    
    # Compute statistics
    two_player_counts = {p: int(two_player_count[p].x) for p in players}
    
    meeting_matrix = np.zeros((n_players, n_players), dtype=int)
    for p1, p2 in combinations(players, 2):
        count = int(meet_count[p1, p2].x)
        meeting_matrix[p1, p2] = count
        meeting_matrix[p2, p1] = count
    
    stats = {
        "status": "optimal" if status == OptimizationStatus.OPTIMAL else "feasible",
        "max_two_player_games": int(max_two_player.x),
        "max_pairwise_meetings": int(max_meet.x),
        "two_player_counts": two_player_counts,
        "meeting_matrix": meeting_matrix,
        "objective_value": model.objective_value,
        "feasibility": feasibility
    }
    
    return schedule, stats


def print_schedule(schedule, stats):
    """Print the schedule in readable format."""
    print(f"=== SOLUTION ===")
    print(f"Status: {stats['status']}")
    print(f"Max 2-player games per player: {stats['max_two_player_games']}")
    print(f"Max pairwise meetings: {stats['max_pairwise_meetings']}")
    print()
    
    for r, round_games in enumerate(schedule):
        print(f"Round {r + 1}:")
        for i, game in enumerate(round_games):
            game_type = "2P" if len(game) == 2 else "3P"
            print(f"  Game {i + 1} ({game_type}): Players {game}")
        print()
    
    print("2-player game distribution:")
    counts = {}
    for p, count in stats['two_player_counts'].items():
        counts[count] = counts.get(count, 0) + 1
    for count in sorted(counts.keys()):
        print(f"  {counts[count]} players with {count} 2P games")
    print()
    
    print("Pairwise meeting statistics:")
    meeting_counts = stats['meeting_matrix'][np.triu_indices_from(stats['meeting_matrix'], k=1)]
    unique, counts = np.unique(meeting_counts, return_counts=True)
    for meetings, count in zip(unique, counts):
        print(f"  {count} pairs met {meetings} time(s)")


def test_corner_cases():
    """Test all corner cases in range 8-20."""
    corner_cases = [8, 10, 11, 13, 14, 16, 17, 19, 20]
    
    for n in corner_cases:
        print(f"\n{'='*60}")
        print(f"Testing n={n} players")
        print('='*60)
        
        schedule, stats = create_game_schedule(n)
        
        if schedule:
            print_schedule(schedule, stats)
            
            # Verify achieved matches theoretical bound
            theoretical = stats['feasibility']['min_max_2p_per_player']
            achieved = stats['max_two_player_games']
            if achieved == theoretical:
                print(f"✓ Achieved theoretical optimum for 2P distribution")
            else:
                print(f"✗ Gap: theoretical={theoretical}, achieved={achieved}")
        else:
            print("✗ No feasible solution found")
        
        print()


if __name__ == "__main__":
    # Test a specific case
    n = 11
    schedule, stats = create_game_schedule(n)
    if schedule:
        print_schedule(schedule, stats)
    
    # Test all corner cases
    # test_corner_cases()
