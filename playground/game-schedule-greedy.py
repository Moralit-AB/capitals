import numpy as np
import random
from itertools import combinations
from collections import defaultdict

def analyze_feasibility(n_players, n_rounds=5):
    """Analyze theoretical bounds for 2-player game distribution."""
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
    min_max_2p = (total_2p_slots + n_players - 1) // n_players
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


def create_game_schedule_greedy(n_players, n_rounds=5, max_pairwise_meetings=2, 
                               max_attempts=100, seed=None):
    """
    Create game schedule using greedy heuristic with backtracking.
    Much faster than ILP for practical use.
    
    Args:
        n_players: Number of players
        n_rounds: Number of rounds (default 5)
        max_pairwise_meetings: Maximum times any pair can meet (default 2)
        max_attempts: Number of complete schedule generation attempts
        seed: Random seed for reproducibility
    
    Returns:
        schedule: List of rounds, each round is list of games
        stats: Dictionary with solution statistics
    """
    
    if seed is not None:
        random.seed(seed)
    
    feasibility = analyze_feasibility(n_players, n_rounds)
    print(f"Feasibility analysis for {n_players} players:")
    print(f"  3P games per round: {feasibility['three_player_games_per_round']}")
    print(f"  2P games per round: {feasibility['two_player_games_per_round']}")
    if feasibility['needs_2p_games']:
        print(f"  Total 2P slots: {feasibility['total_2p_slots']}")
        print(f"  Theoretical min max_2p: {feasibility['min_max_2p_per_player']}")
        print(f"  Perfect balance possible: {feasibility['perfect_balance']}")
    print(f"  Max pairwise meetings constraint: {max_pairwise_meetings}")
    print()
    
    best_schedule = None
    best_score = float('inf')
    best_stats = None
    
    for attempt in range(max_attempts):
        schedule, stats = _generate_schedule_attempt(
            n_players, n_rounds, feasibility, max_pairwise_meetings
        )
        
        if schedule is not None:
            # Score: prioritize no violations, then 2P balance, then meeting spread
            violations = len(stats['constraint_violations'])
            max_2p = stats['max_two_player_games']
            max_meetings = stats['max_pairwise_meetings']
            
            score = (violations * 1000000 + max_2p * 1000 + max_meetings)
            
            if score < best_score:
                best_schedule = schedule
                best_score = score
                best_stats = stats
                
                if violations == 0:
                    print(f"✓ Found valid schedule (attempt {attempt + 1}/{max_attempts})")
                    break
        
        if (attempt + 1) % 10 == 0:
            print(f"  Attempt {attempt + 1}/{max_attempts}...")
    
    if best_schedule is None:
        print("✗ Failed to generate valid schedule")
        return None, {"status": "failed", "feasibility": feasibility}
    
    violations = best_stats['constraint_violations']
    if violations:
        print(f"⚠️  Best schedule has {len(violations)} violations")
    else:
        print(f"✓ Valid schedule generated with no violations")
    
    return best_schedule, best_stats


def _generate_schedule_attempt(n_players, n_rounds, feasibility, max_pairwise_meetings):
    """Single attempt to generate a schedule using greedy approach."""
    
    # Track state
    meeting_count = defaultdict(int)  # (p1, p2) -> count
    two_player_count = defaultdict(int)  # player -> count of 2P games
    schedule = []
    
    players = list(range(n_players))
    
    for round_num in range(n_rounds):
        round_games = _generate_round(
            players, feasibility, meeting_count, two_player_count, 
            max_pairwise_meetings, round_num
        )
        
        if round_games is None:
            return None, None
        
        schedule.append(round_games)
    
    # Compute statistics
    actual_meetings = {}
    for r, round_games in enumerate(schedule):
        for game in round_games:
            for i, p1 in enumerate(game):
                for p2 in game[i+1:]:
                    pair = (min(p1, p2), max(p1, p2))
                    actual_meetings[pair] = actual_meetings.get(pair, 0) + 1
    
    violations = [(p1, p2, count) for (p1, p2), count in actual_meetings.items() 
                  if count > max_pairwise_meetings]
    
    meeting_matrix = np.zeros((n_players, n_players), dtype=int)
    pairwise_details = {}
    for (p1, p2), count in actual_meetings.items():
        meeting_matrix[p1, p2] = count
        meeting_matrix[p2, p1] = count
        pairwise_details[(p1, p2)] = count
    
    for p1, p2 in combinations(range(n_players), 2):
        if (p1, p2) not in pairwise_details:
            pairwise_details[(p1, p2)] = 0
    
    stats = {
        "status": "optimal" if not violations else "feasible",
        "max_two_player_games": max(two_player_count.values()) if two_player_count else 0,
        "max_pairwise_meetings": max(actual_meetings.values()) if actual_meetings else 0,
        "two_player_counts": dict(two_player_count),
        "meeting_matrix": meeting_matrix,
        "pairwise_details": pairwise_details,
        "feasibility": feasibility,
        "constraint_violations": violations
    }
    
    return schedule, stats


def _generate_round(players, feasibility, meeting_count, two_player_count, 
                    max_pairwise_meetings, round_num):
    """Generate games for a single round using greedy approach."""
    
    n_3p = feasibility['three_player_games_per_round']
    n_2p = feasibility['two_player_games_per_round']
    
    available = players.copy()
    random.shuffle(available)
    
    round_games = []
    
    # Generate 3-player games first
    for _ in range(n_3p):
        if len(available) < 3:
            return None
        
        game = _select_best_game(available, 3, meeting_count, max_pairwise_meetings)
        if game is None:
            return None
        
        round_games.append(game)
        for p in game:
            available.remove(p)
        
        # Update meeting counts
        for i, p1 in enumerate(game):
            for p2 in game[i+1:]:
                pair = (min(p1, p2), max(p1, p2))
                meeting_count[pair] += 1
    
    # Generate 2-player games
    for _ in range(n_2p):
        if len(available) < 2:
            return None
        
        game = _select_best_game(available, 2, meeting_count, max_pairwise_meetings,
                                two_player_count=two_player_count)
        if game is None:
            return None
        
        round_games.append(game)
        for p in game:
            available.remove(p)
            two_player_count[p] += 1
        
        # Update meeting counts
        for i, p1 in enumerate(game):
            for p2 in game[i+1:]:
                pair = (min(p1, p2), max(p1, p2))
                meeting_count[pair] += 1
    
    return round_games


def _select_best_game(available, game_size, meeting_count, max_pairwise_meetings,
                     two_player_count=None):
    """
    Select best game of given size from available players.
    Prioritizes: 
    1. Not violating pairwise meeting constraint
    2. Minimizing max meetings in this game
    3. Balancing 2P game distribution (if applicable)
    """
    
    if len(available) < game_size:
        return None
    
    # Try multiple random samples to find a good game
    best_game = None
    best_score = float('inf')
    
    n_samples = min(100, len(list(combinations(available, game_size))))
    
    if n_samples <= 20:
        # Enumerate all possibilities if few enough
        candidates = list(combinations(available, game_size))
    else:
        # Sample randomly
        candidates = []
        for _ in range(n_samples):
            sample = random.sample(available, game_size)
            candidates.append(tuple(sorted(sample)))
        candidates = list(set(candidates))  # Remove duplicates
    
    for game in candidates:
        # Check if this game would violate constraints
        pairs = [(min(game[i], game[j]), max(game[i], game[j])) 
                 for i in range(len(game)) for j in range(i+1, len(game))]
        
        # Count violations and max meetings
        violations = sum(1 for pair in pairs if meeting_count[pair] >= max_pairwise_meetings)
        max_meetings = max((meeting_count[pair] for pair in pairs), default=0)
        total_meetings = sum(meeting_count[pair] for pair in pairs)
        
        # For 2P games, consider balance
        two_p_imbalance = 0
        if two_player_count is not None and game_size == 2:
            two_p_imbalance = sum(two_player_count[p] for p in game)
        
        score = (violations * 100000 + max_meetings * 1000 + 
                total_meetings * 10 + two_p_imbalance)
        
        if score < best_score:
            best_score = score
            best_game = list(game)
    
    # If all games violate, return None to trigger backtrack
    if best_game and best_score >= 100000:
        return None
    
    return best_game


def print_schedule(schedule, stats, show_all_pairs=False):
    """Print the schedule in readable format."""
    print(f"=== SOLUTION ===")
    print(f"Status: {stats['status']}")
    print(f"Max 2-player games per player: {stats['max_two_player_games']}")
    print(f"Max pairwise meetings: {stats['max_pairwise_meetings']}")
    
    if stats.get('constraint_violations'):
        print(f"⚠️  CONSTRAINT VIOLATIONS: {len(stats['constraint_violations'])} pairs exceed limit")
    else:
        print(f"✓ All constraints satisfied")
    
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
    if show_all_pairs:
        print("Detailed pairwise encounters:")
        pairwise_by_count = {}
        for (p1, p2), count in sorted(stats['pairwise_details'].items()):
            if count not in pairwise_by_count:
                pairwise_by_count[count] = []
            pairwise_by_count[count].append((p1, p2))
        
        for count in sorted(pairwise_by_count.keys(), reverse=True):
            pairs = pairwise_by_count[count]
            print(f"\n  Pairs that met {count} time(s): ({len(pairs)} pairs)")
            if count >= stats['max_pairwise_meetings'] or len(pairs) <= 20:
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
    
    encounter_matrix = np.zeros((n_players, n_players), dtype=int)
    
    for r, round_games in enumerate(schedule):
        for game in round_games:
            for i, p1 in enumerate(game):
                for p2 in game[i+1:]:
                    encounter_matrix[p1, p2] += 1
                    encounter_matrix[p2, p1] += 1
    
    print("Encounter matrix (manual count):")
    print(encounter_matrix)
    print()
    
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
        
        schedule, stats = create_game_schedule_greedy(n, max_attempts=50)
        
        if schedule:
            print_schedule(schedule, stats, show_all_pairs=False)
            verify_schedule(schedule, n)
            
            theoretical = stats['feasibility']['min_max_2p_per_player']
            achieved = stats['max_two_player_games']
            if achieved == theoretical:
                print(f"✓ Achieved theoretical optimum for 2P distribution")
            else:
                print(f"✗ Gap: theoretical={theoretical}, achieved={achieved}")
            
            if not stats.get('constraint_violations'):
                print(f"✓ All pairwise constraints satisfied")
        else:
            print(f"✗ No feasible solution found")
        
        print()


if __name__ == "__main__":
    # Test a specific case
    n = 11
    schedule, stats = create_game_schedule_greedy(n, max_pairwise_meetings=2, 
                                                  max_attempts=1000, seed=42)
    if schedule:
        print_schedule(schedule, stats, show_all_pairs=True)
        verify_schedule(schedule, n)
    
    # Uncomment to test all corner cases
    # test_corner_cases()