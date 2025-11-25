from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary, LpInteger, LpStatus, value
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
            "three_player_games_per_round": n_players // 3,
            "two_player_games_per_round": 0,
            "total_2p_slots": 0,
            "min_max_2p_per_player": 0,
            "perfect_balance": True
        }
    
    if remainder == 1:
        three_player_games = (n_players - 4) // 3
        two_player_games = 2
    else:  # remainder == 2
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
    Create optimal game schedule using ILP with PuLP.
    
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
    print(f"  3P games per round: {feasibility['three_player_games_per_round']}")
    print(f"  2P games per round: {feasibility['two_player_games_per_round']}")
    if feasibility['needs_2p_games']:
        print(f"  Total 2P slots: {feasibility['total_2p_slots']}")
        print(f"  Theoretical min max_2p: {feasibility['min_max_2p_per_player']}")
        print(f"  Perfect balance possible: {feasibility['perfect_balance']}")
    print()
    
    model = LpProblem("game_schedule", LpMinimize)
    
    players = range(n_players)
    rounds = range(n_rounds)
    games = range(max_games_per_round)
    
    # Decision variables
    x = {}
    for p in players:
        for g in games:
            for r in rounds:
                x[p, g, r] = LpVariable(f"x_{p}_{g}_{r}", cat=LpBinary)
    
    y = {}
    for g in games:
        for r in rounds:
            y[g, r] = LpVariable(f"y_{g}_{r}", cat=LpBinary)
    
    game_used = {}
    for g in games:
        for r in rounds:
            game_used[g, r] = LpVariable(f"used_{g}_{r}", cat=LpBinary)
    
    # Auxiliary variable for linearizing x * y product
    z = {}
    for p in players:
        for g in games:
            for r in rounds:
                z[p, g, r] = LpVariable(f"z_{p}_{g}_{r}", cat=LpBinary)
    
    two_player_count = {}
    for p in players:
        two_player_count[p] = LpVariable(f"two_count_{p}", lowBound=0, cat=LpInteger)
    
    max_two_player = LpVariable("max_two_player", lowBound=0, cat=LpInteger)
    
    meet_count = {}
    for p1, p2 in combinations(players, 2):
        meet_count[p1, p2] = LpVariable(f"meet_{p1}_{p2}", lowBound=0, cat=LpInteger)
    
    max_meet = LpVariable("max_meet", lowBound=0, cat=LpInteger)
    
    # Constraints
    
    # 1. Each player in exactly one game per round
    for r in rounds:
        for p in players:
            model += lpSum(x[p, g, r] for g in games) == 1, f"player_{p}_round_{r}"
    
    # 2. Game size is 2 or 3 players (only if game is used)
    for g in games:
        for r in rounds:
            game_size = lpSum(x[p, g, r] for p in players)
            model += game_size >= 2 * game_used[g, r], f"game_min_{g}_{r}"
            model += game_size <= 3 * game_used[g, r], f"game_max_{g}_{r}"
    
    # 3. Link y to game size: y[g,r]=1 iff game g in round r has 2 players
    for g in games:
        for r in rounds:
            game_size = lpSum(x[p, g, r] for p in players)
            model += game_size >= 2 * y[g, r], f"two_player_min_{g}_{r}"
            model += game_size <= 2 + (1 - y[g, r]), f"two_player_max_{g}_{r}"
    
    # 3b. Linearize z[p,g,r] = x[p,g,r] * y[g,r]
    for p in players:
        for g in games:
            for r in rounds:
                model += z[p, g, r] <= x[p, g, r], f"lin1_{p}_{g}_{r}"
                model += z[p, g, r] <= y[g, r], f"lin2_{p}_{g}_{r}"
                model += z[p, g, r] >= x[p, g, r] + y[g, r] - 1, f"lin3_{p}_{g}_{r}"
    
    # 4. Count 2-player games per player using linearized z variable
    for p in players:
        model += two_player_count[p] == lpSum(z[p, g, r] 
                                               for g in games for r in rounds), f"count_2p_{p}"
    
    # 5. Bound max_two_player
    for p in players:
        model += max_two_player >= two_player_count[p], f"max_2p_{p}"
    
    # 6. Count pairwise meetings
    for r in rounds:
        for g in games:
            for p1, p2 in combinations(players, 2):
                model += meet_count[p1, p2] >= x[p1, g, r] + x[p2, g, r] - 1, f"meet_{p1}_{p2}_{g}_{r}"
    
    # 7. Bound max_meet
    for p1, p2 in combinations(players, 2):
        model += max_meet >= meet_count[p1, p2], f"max_meet_{p1}_{p2}"
    
    # 8. Game ordering (symmetry breaking): use lower-indexed games first
    for g in range(len(games) - 1):
        for r in rounds:
            model += game_used[g, r] >= game_used[g + 1, r], f"symmetry_{g}_{r}"
    
    # 9. Constrain exact number of 2-player AND 3-player games per round
    for r in rounds:
        model += lpSum(y[g, r] for g in games) == feasibility['two_player_games_per_round'], f"two_player_count_round_{r}"
        model += lpSum(game_used[g, r] - y[g, r] for g in games) == feasibility['three_player_games_per_round'], f"three_player_count_round_{r}"
    
    # Objective: Lexicographic - prioritize 2-player fairness, then meetings
    model += 1000 * max_two_player + max_meet, "objective"
    
    # Solve
    print("Solving... (this may take a minute)")
    model.solve()
    
    status = LpStatus[model.status]
    
    if status not in ["Optimal", "Not Solved"]:
        return None, {"status": status.lower(), "feasibility": feasibility}
    
    # Extract solution
    schedule = []
    for r in rounds:
        round_games = []
        for g in games:
            game_players = [p for p in players if value(x[p, g, r]) >= 0.99]
            if len(game_players) >= 2:
                round_games.append(sorted(game_players))
        schedule.append(round_games)
    
    # Compute statistics
    two_player_counts_dict = {p: int(value(two_player_count[p])) for p in players}
    
    meeting_matrix = np.zeros((n_players, n_players), dtype=int)
    pairwise_details = {}
    for p1, p2 in combinations(players, 2):
        count = int(value(meet_count[p1, p2]))
        meeting_matrix[p1, p2] = count
        meeting_matrix[p2, p1] = count
        pairwise_details[(p1, p2)] = count
    
    stats = {
        "status": status.lower(),
        "max_two_player_games": int(value(max_two_player)),
        "max_pairwise_meetings": int(value(max_meet)),
        "two_player_counts": two_player_counts_dict,
        "meeting_matrix": meeting_matrix,
        "pairwise_details": pairwise_details,
        "objective_value": value(model.objective),
        "feasibility": feasibility
    }
    
    return schedule, stats


def print_schedule(schedule, stats, show_all_pairs=False):
    """Print the schedule in readable format."""
    print(f"=== SOLUTION ===")
    print(f"Status: {stats['status']}")
    print(f"Max 2-player games per player: {stats['max_two_player_games']}")
    print(f"Max pairwise meetings: {stats['max_pairwise_meetings']}")
    print()
    
    for r, round_games in enumerate(schedule):
        print(f"Round {r + 1}:")
        three_p_count = sum(1 for game in round_games if len(game) == 3)
        two_p_count = sum(1 for game in round_games if len(game) == 2)
        print(f"  ({three_p_count}×3P, {two_p_count}×2P)")
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
    
    print("Pairwise meeting statistics (summary):")
    meeting_counts = stats['meeting_matrix'][np.triu_indices_from(stats['meeting_matrix'], k=1)]
    unique, counts_array = np.unique(meeting_counts, return_counts=True)
    for meetings, count in zip(unique, counts_array):
        print(f"  {count} pairs met {meetings} time(s)")
    print()
    
    # Show detailed pairwise encounters
    print("Detailed pairwise encounters:")
    pairwise_by_count = {}
    for (p1, p2), count in sorted(stats['pairwise_details'].items()):
        if count not in pairwise_by_count:
            pairwise_by_count[count] = []
        pairwise_by_count[count].append((p1, p2))
    
    for count in sorted(pairwise_by_count.keys(), reverse=True):
        pairs = pairwise_by_count[count]
        print(f"\n  Pairs that met {count} time(s): ({len(pairs)} pairs)")
        if show_all_pairs or count >= stats['max_pairwise_meetings']:
            for p1, p2 in sorted(pairs):
                print(f"    Player {p1} - Player {p2}")
        elif len(pairs) <= 20:
            for p1, p2 in sorted(pairs):
                print(f"    Player {p1} - Player {p2}")
        else:
            print(f"    (showing first 10 of {len(pairs)})")
            for p1, p2 in sorted(pairs)[:10]:
                print(f"    Player {p1} - Player {p2}")
            print(f"    ...")
    print()


def verify_schedule(schedule, n_players):
    """Manually verify the schedule and count encounters."""
    print("=== VERIFICATION ===")
    
    # Count encounters manually
    encounter_matrix = np.zeros((n_players, n_players), dtype=int)
    
    for r, round_games in enumerate(schedule):
        for game in round_games:
            # All players in a game meet each other
            for i, p1 in enumerate(game):
                for p2 in game[i+1:]:
                    encounter_matrix[p1, p2] += 1
                    encounter_matrix[p2, p1] += 1
    
    # Display verification
    print("Encounter matrix (manual count):")
    print(encounter_matrix)
    print()
    
    # Count distribution
    upper_tri = encounter_matrix[np.triu_indices_from(encounter_matrix, k=1)]
    unique, counts = np.unique(upper_tri, return_counts=True)
    print("Encounter distribution (verification):")
    for meetings, count in zip(unique, counts):
        print(f"  {count} pairs met {meetings} time(s)")
    print()
    
    return encounter_matrix


def test_corner_cases():
    """Test all corner cases in range 8-20."""
    corner_cases = [8, 10, 11, 13, 14, 16, 17, 19, 20]
    
    for n in corner_cases:
        print(f"\n{'='*60}")
        print(f"Testing n={n} players")
        print('='*60)
        
        schedule, stats = create_game_schedule(n)
        
        if schedule:
            print_schedule(schedule, stats, show_all_pairs=False)
            
            # Verify manually
            verify_schedule(schedule, n)
            
            # Check theoretical bound
            theoretical = stats['feasibility']['min_max_2p_per_player']
            achieved = stats['max_two_player_games']
            if achieved == theoretical:
                print(f"✓ Achieved theoretical optimum for 2P distribution")
            else:
                print(f"✗ Gap: theoretical={theoretical}, achieved={achieved}")
        else:
            print(f"✗ No feasible solution found (status: {stats['status']})")
        
        print()


if __name__ == "__main__":
    # Test a specific case
    n = 11
    schedule, stats = create_game_schedule(n)
    if schedule:
        print_schedule(schedule, stats, show_all_pairs=True)
        verify_schedule(schedule, n)
    
    # Uncomment to test all corner cases
    # test_corner_cases()
