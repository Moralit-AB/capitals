import numpy as np
import random
from itertools import combinations
from collections import defaultdict
import time

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
                               max_attempts=100, verbose=True):
    """
    Create game schedule using greedy heuristic with strong 2P fairness prioritization.
    
    Args:
        n_players: Number of players
        n_rounds: Number of rounds (default 5)
        max_pairwise_meetings: Maximum times any pair can meet (default 2)
        max_attempts: Number of complete schedule generation attempts
        verbose: Whether to print progress
    
    Returns:
        schedule: List of rounds, each round is list of games
        stats: Dictionary with solution statistics
    """
    
    feasibility = analyze_feasibility(n_players, n_rounds)
    
    if verbose:
        print(f"Feasibility analysis for {n_players} players:")
        print(f"  3P games per round: {feasibility['three_player_games_per_round']}")
        print(f"  2P games per round: {feasibility['two_player_games_per_round']}")
        if feasibility['needs_2p_games']:
            print(f"  Total 2P slots: {feasibility['total_2p_slots']}")
            print(f"  Theoretical min max_2p: {feasibility['min_max_2p_per_player']}")
            print(f"  Perfect balance possible: {feasibility['perfect_balance']}")
        print(f"  Max pairwise meetings constraint: {max_pairwise_meetings}")
        print(f"  Attempting {max_attempts} schedule generations...")
        print()
    
    start_time = time.time()
    
    best_schedule = None
    best_score = float('inf')
    best_stats = None
    
    # Track when we find optimal solutions
    optimal_found_at = None
    theoretical_2p = feasibility['min_max_2p_per_player']
    
    # Early stopping if we find optimal quickly
    optimal_found_count = 0
    
    for attempt in range(max_attempts):
        schedule, stats = _generate_schedule_attempt(
            n_players, n_rounds, feasibility, max_pairwise_meetings
        )
        
        if schedule is not None:
            # Score: STRONGLY prioritize 2P fairness first, then violations, then meeting spread
            violations = len(stats['constraint_violations'])
            max_2p = stats['max_two_player_games']
            max_meetings = stats['max_pairwise_meetings']
            
            # Weight max_2p very heavily (1,000,000) so it dominates
            score = (max_2p * 1000000 + violations * 10000 + max_meetings)
            
            if score < best_score:
                best_schedule = schedule
                best_score = score
                best_stats = stats
                
                if verbose and (attempt < 10 or (attempt + 1) % 10 == 0):
                    print(f"  Attempt {attempt + 1}: New best! max_2p={max_2p}, violations={violations}, max_meetings={max_meetings}")
                
                # Check if optimal
                if violations == 0 and max_2p == theoretical_2p:
                    if optimal_found_at is None:
                        optimal_found_at = attempt + 1
                    optimal_found_count += 1
                    
                    # Early stopping: if we found optimal 5 times in a row, likely won't improve
                    if optimal_found_count >= 5:
                        if verbose:
                            print(f"\n✓ Found optimal solution 5 times, stopping early at attempt {attempt + 1}")
                        break
            else:
                # Reset optimal counter if we didn't find optimal
                if stats and (len(stats['constraint_violations']) > 0 or 
                            stats['max_two_player_games'] > theoretical_2p):
                    optimal_found_count = 0
        
        if verbose and (attempt + 1) % 50 == 0 and attempt + 1 != max_attempts:
            elapsed = time.time() - start_time
            print(f"  Progress: {attempt + 1}/{max_attempts} attempts ({elapsed:.1f}s), best max_2p={best_stats['max_two_player_games'] if best_stats else 'N/A'}")
    
    elapsed_time = time.time() - start_time
    
    if best_schedule is None:
        if verbose:
            print("✗ Failed to generate valid schedule")
        return None, {"status": "failed", "feasibility": feasibility}
    
    violations = best_stats['constraint_violations']
    max_2p = best_stats['max_two_player_games']
    
    if verbose:
        print(f"\n=== SEARCH COMPLETE ===")
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Attempts: {attempt + 1}")
        
        if violations:
            print(f"⚠️  Best schedule has {len(violations)} pairwise violations")
        else:
            print(f"✓ Valid schedule with no pairwise violations")
        
        if max_2p == theoretical_2p:
            print(f"✓ Achieved optimal 2P distribution (max {max_2p} per player)")
            if optimal_found_at:
                print(f"  First found at attempt {optimal_found_at}")
        else:
            print(f"⚠️  2P distribution: max {max_2p} per player (theoretical optimum: {theoretical_2p})")
        print()
    
    best_stats['search_time'] = elapsed_time
    best_stats['attempts_used'] = attempt + 1
    
    return best_schedule, best_stats


def _generate_schedule_attempt(n_players, n_rounds, feasibility, max_pairwise_meetings):
    """Single attempt to generate a schedule using greedy approach with 2P fairness."""
    
    # Track state
    meeting_count = defaultdict(int)
    two_player_count = defaultdict(int)
    schedule = []
    
    players = list(range(n_players))
    
    for round_num in range(n_rounds):
        round_games = _generate_round_2p_fair(
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


def _generate_round_2p_fair(players, feasibility, meeting_count, two_player_count, 
                            max_pairwise_meetings, round_num):
    """Generate games for a single round with STRONG 2P fairness priority."""
    
    n_3p = feasibility['three_player_games_per_round']
    n_2p = feasibility['two_player_games_per_round']
    
    available = players.copy()
    round_games = []
    
    # STEP 1: Generate 2-player games FIRST
    for _ in range(n_2p):
        if len(available) < 2:
            return None
        
        game = _select_2p_game_fair(available, meeting_count, two_player_count, 
                                     max_pairwise_meetings)
        if game is None:
            return None
        
        round_games.append(game)
        for p in game:
            available.remove(p)
            two_player_count[p] += 1
        
        pair = (min(game[0], game[1]), max(game[0], game[1]))
        meeting_count[pair] += 1
    
    # STEP 2: Generate 3-player games
    for _ in range(n_3p):
        if len(available) < 3:
            return None
        
        game = _select_3p_game(available, meeting_count, max_pairwise_meetings)
        if game is None:
            return None
        
        round_games.append(game)
        for p in game:
            available.remove(p)
        
        for i, p1 in enumerate(game):
            for p2 in game[i+1:]:
                pair = (min(p1, p2), max(p1, p2))
                meeting_count[pair] += 1
    
    return round_games


def _select_2p_game_fair(available, meeting_count, two_player_count, max_pairwise_meetings):
    """Select 2-player game prioritizing players with fewest 2P games so far."""
    
    if len(available) < 2:
        return None
    
    by_2p_count = defaultdict(list)
    for p in available:
        by_2p_count[two_player_count[p]].append(p)
    
    sorted_counts = sorted(by_2p_count.keys())
    
    best_game = None
    best_score = float('inf')
    
    for max_count_allowed in sorted_counts:
        candidate_players = []
        for count in sorted_counts:
            if count <= max_count_allowed:
                candidate_players.extend(by_2p_count[count])
        
        if len(candidate_players) < 2:
            continue
        
        for p1, p2 in combinations(candidate_players, 2):
            pair = (min(p1, p2), max(p1, p2))
            
            current_meetings = meeting_count[pair]
            if current_meetings >= max_pairwise_meetings:
                continue
            
            max_2p_in_pair = max(two_player_count[p1], two_player_count[p2])
            score = (max_2p_in_pair * 1000 + current_meetings)
            
            if score < best_score:
                best_score = score
                best_game = [p1, p2]
        
        if best_game is not None:
            return best_game
    
    if best_game is None:
        for p1, p2 in combinations(available, 2):
            pair = (min(p1, p2), max(p1, p2))
            if meeting_count[pair] < max_pairwise_meetings:
                return [p1, p2]
    
    return best_game


def _select_3p_game(available, meeting_count, max_pairwise_meetings):
    """Select 3-player game minimizing pairwise meetings."""
    
    if len(available) < 3:
        return None
    
    best_game = None
    best_score = float('inf')
    
    n_samples = min(100, len(list(combinations(available, 3))))
    
    if n_samples <= 30:
        candidates = list(combinations(available, 3))
    else:
        candidates = set()
        for _ in range(n_samples):
            sample = tuple(sorted(random.sample(available, 3)))
            candidates.add(sample)
        candidates = list(candidates)
    
    for game in candidates:
        pairs = [(min(game[i], game[j]), max(game[i], game[j])) 
                 for i in range(3) for j in range(i+1, 3)]
        
        violations = sum(1 for pair in pairs if meeting_count[pair] >= max_pairwise_meetings)
        
        if violations > 0:
            continue
        
        max_meetings = max(meeting_count[pair] for pair in pairs)
        total_meetings = sum(meeting_count[pair] for pair in pairs)
        
        score = (max_meetings * 100 + total_meetings)
        
        if score < best_score:
            best_score = score
            best_game = list(game)
    
    return best_game


def compare_multiple_runs(n_players, n_rounds=5, max_pairwise_meetings=2, 
                         num_runs=10, attempts_per_run=10):
    """
    Run the scheduler multiple times and compare results.
    Useful for understanding solution quality distribution.
    """
    
    print(f"=== COMPARING {num_runs} INDEPENDENT RUNS ===")
    print(f"Each run attempts {attempts_per_run} schedules\n")
    
    results = []
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}...")
        schedule, stats = create_game_schedule_greedy(
            n_players, n_rounds, max_pairwise_meetings, 
            max_attempts=attempts_per_run, verbose=False
        )
        
        if schedule:
            results.append({
                'max_2p': stats['max_two_player_games'],
                'violations': len(stats['constraint_violations']),
                'max_meetings': stats['max_pairwise_meetings'],
                'schedule': schedule,
                'stats': stats
            })
    
    if not results:
        print("✗ No valid schedules found")
        return None
    
    print(f"\n=== COMPARISON RESULTS ===")
    print(f"Successful runs: {len(results)}/{num_runs}")
    
    # Analyze distribution
    max_2p_values = [r['max_2p'] for r in results]
    violations_values = [r['violations'] for r in results]
    max_meetings_values = [r['max_meetings'] for r in results]
    
    print(f"\nMax 2P games per player:")
    print(f"  Best: {min(max_2p_values)}")
    print(f"  Worst: {max(max_2p_values)}")
    print(f"  Average: {np.mean(max_2p_values):.2f}")
    print(f"  Distribution: {dict((x, max_2p_values.count(x)) for x in set(max_2p_values))}")
    
    print(f"\nPairwise constraint violations:")
    print(f"  Best: {min(violations_values)}")
    print(f"  Worst: {max(violations_values)}")
    print(f"  Runs with 0 violations: {violations_values.count(0)}/{len(results)}")
    
    print(f"\nMax pairwise meetings:")
    print(f"  Best: {min(max_meetings_values)}")
    print(f"  Worst: {max(max_meetings_values)}")
    
    # Return best solution
    best_result = min(results, key=lambda r: (r['max_2p'], r['violations'], r['max_meetings']))
    print(f"\nBest overall solution:")
    print(f"  Max 2P: {best_result['max_2p']}")
    print(f"  Violations: {best_result['violations']}")
    print(f"  Max meetings: {best_result['max_meetings']}")
    
    return best_result['schedule'], best_result['stats']


def print_schedule(schedule, stats, show_all_pairs=False):
    """Print the schedule in readable format."""
    print(f"\n=== SOLUTION ===")
    print(f"Status: {stats['status']}")
    if 'search_time' in stats:
        print(f"Search time: {stats['search_time']:.2f}s ({stats['attempts_used']} attempts)")
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


if __name__ == "__main__":
    n = 11
    
    # Option 1: Single run with many attempts (recommended)
    print("=== STRATEGY 1: Single run with 100 attempts ===\n")
    # It seems the model is quite dependent on initial conditions, because even with
    # a high number of max_attempts it sometimes fails to find the better solution.
    # Running with ~100 max_attempts and doing so multiple times increases the likelihood
    # of finding a good solution.
    schedule1, stats1 = create_game_schedule_greedy(n, max_pairwise_meetings=2, 
                                                     max_attempts=1000)
    if schedule1:
        print_schedule(schedule1, stats1)
    
    print("\n" + "="*60 + "\n")
    
    # # Option 2: Multiple independent runs (for comparison)
    # print("=== STRATEGY 2: 10 independent runs with 10 attempts each ===\n")
    # schedule2, stats2 = compare_multiple_runs(n, max_pairwise_meetings=2,
    #                                          num_runs=10, attempts_per_run=10)
    # if schedule2:
    #     print_schedule(schedule2, stats2)