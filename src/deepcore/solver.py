"""
solver.py
Student implementation file for DeepCore search algorithms.
"""

from collections import deque
import heapq
import itertools
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .environment import DeepCoreEnv
from .state import DeepCoreState, SearchMetrics


def _reconstruct_path(
        parent_map: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]],
        goal_state: DeepCoreState,
) -> List[str]:
    actions: List[str] = []
    curr: Optional[DeepCoreState] = goal_state
    while curr is not None:
        prev, act = parent_map.get(curr, (None, None))
        if act is not None:
            actions.append(act)
        curr = prev
    actions.reverse()
    return actions


def _unpack_chain(chain: Optional[Tuple[Any, str]]) -> List[str]:
    actions: List[str] = []
    curr = chain
    while curr is not None:
        prev, act = curr
        actions.append(act)
        curr = prev
    actions.reverse()
    return actions


# =============================================================================
# 1. Breadth-First Search (BFS)
# =============================================================================
def breadth_first_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="BFS")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics

    frontier = deque([start])
    frontier_set = {start}
    explored: Set[DeepCoreState] = set()
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
    cost_so_far: Dict[DeepCoreState, int] = {start: 0}

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))
        state = frontier.popleft()
        frontier_set.discard(state)

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            if next_state not in explored and next_state not in frontier_set:
                parent[next_state] = (state, action.value)
                cost_so_far[next_state] = cost_so_far[state] + step_cost
                if env.is_goal(next_state):
                    metrics.solved = True
                    metrics.solution_actions = _reconstruct_path(parent, next_state)
                    metrics.total_cost = cost_so_far[next_state]
                    return metrics
                frontier.append(next_state)
                frontier_set.add(next_state)
                metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# 2. Depth-First Search (DFS) & Iterative Deepening Search (IDS)
# =============================================================================
def depth_first_search(env: DeepCoreEnv, max_expansions: Optional[int] = None) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="DFS")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics

    frontier: List[Tuple[DeepCoreState, Optional[Tuple[Any, str]], int]] = [(start, None, 0)]
    explored: Set[DeepCoreState] = set()

    while frontier:
        if max_expansions is not None and metrics.nodes_expanded >= max_expansions:
            metrics.status_message = "Expansion limit reached."
            return metrics

        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

        state, chain, g = frontier.pop()
        if state in explored:
            continue

        if env.is_goal(state):
            metrics.solved = True
            metrics.solution_actions = _unpack_chain(chain)
            metrics.total_cost = g
            return metrics

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            if next_state not in explored:
                frontier.append((next_state, (chain, action.value), g + step_cost))

    metrics.status_message = "No solution found."
    return metrics


def iterative_deepening_search(
        env: DeepCoreEnv,
        max_depth: Optional[int] = None,
        max_expansions: Optional[int] = None,
) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="IDS")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics

    best_depth: Dict[DeepCoreState, int] = {start: 0}
    limit = 0

    while True:
        if max_depth is not None and limit > max_depth:
            metrics.status_message = "No solution found."
            return metrics

        frontier: List[Tuple[DeepCoreState, Optional[Tuple[Any, str]], int, int]] = [(start, None, 0, 0)]
        visited_in_iter: Set[DeepCoreState] = {start}
        cut_off_occurred = False

        while frontier:
            if max_expansions is not None and metrics.nodes_expanded >= max_expansions:
                metrics.status_message = "Expansion limit reached."
                return metrics

            metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
            metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(visited_in_iter))

            state, chain, depth, cost = frontier.pop()

            if env.is_goal(state):
                metrics.solved = True
                metrics.solution_actions = _unpack_chain(chain)
                metrics.total_cost = cost
                return metrics

            if depth == limit:
                cut_off_occurred = True
                continue

            metrics.nodes_expanded += 1
            metrics.explored_order.append(state.bot_pos)

            for action, next_state, step_cost in env.get_successors(state):
                metrics.nodes_generated += 1
                new_depth = depth + 1
                if next_state not in visited_in_iter and best_depth.get(next_state, new_depth) == new_depth:
                    best_depth[next_state] = new_depth
                    visited_in_iter.add(next_state)
                    frontier.append((next_state, (chain, action.value), new_depth, cost + step_cost))

        if not cut_off_occurred:
            metrics.status_message = "No solution found."
            return metrics

        limit += 1


# =============================================================================
# 3. Bidirectional Search
# =============================================================================
def bidirectional_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="Bidirectional")
    start = env.get_initial_state()
    goals = set(env.get_goal_states())

    if start in goals or env.is_goal(start):
        metrics.solved = True
        return metrics

    frontier_f = deque([start])
    frontier_b = deque(goals)
    frontier_set_f = {start}
    frontier_set_b = set(goals)
    explored_f: Set[DeepCoreState] = set()
    explored_b: Set[DeepCoreState] = set()
    parent_f: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
    parent_b: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {g: (None, None) for g in goals}
    cost_f: Dict[DeepCoreState, int] = {start: 0}
    cost_b: Dict[DeepCoreState, int] = {g: 0 for g in goals}

    def build_path(meet_node: DeepCoreState) -> List[str]:
        path = _reconstruct_path(parent_f, meet_node)
        curr: Optional[DeepCoreState] = meet_node
        while curr is not None:
            nxt, act = parent_b.get(curr, (None, None))
            if act is not None:
                path.append(act)
            curr = nxt
        return path

    while frontier_f or frontier_b:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier_f) + len(frontier_b))
        metrics.max_memory_size = max(
            metrics.max_memory_size,
            len(frontier_f) + len(frontier_b) + len(explored_f) + len(explored_b),
        )

        if frontier_f:
            curr_f = frontier_f.popleft()
            frontier_set_f.discard(curr_f)
            explored_f.add(curr_f)
            metrics.nodes_expanded += 1
            metrics.explored_order.append(curr_f.bot_pos)

            for act_f, nxt_f, cost_step_f in env.get_successors(curr_f):
                metrics.nodes_generated += 1
                if nxt_f not in explored_f and nxt_f not in frontier_set_f:
                    parent_f[nxt_f] = (curr_f, act_f.value)
                    cost_f[nxt_f] = cost_f[curr_f] + cost_step_f
                    if nxt_f in frontier_set_b or nxt_f in explored_b or env.is_goal(nxt_f):
                        metrics.solved = True
                        metrics.solution_actions = build_path(nxt_f)
                        metrics.total_cost = cost_f[nxt_f] + cost_b.get(nxt_f, 0)
                        return metrics
                    frontier_f.append(nxt_f)
                    frontier_set_f.add(nxt_f)

        if frontier_b:
            curr_b = frontier_b.popleft()
            frontier_set_b.discard(curr_b)
            explored_b.add(curr_b)
            metrics.nodes_expanded += 1

            for act_b, prev_b, cost_step_b in env.get_predecessors(curr_b):
                metrics.nodes_generated += 1
                if prev_b not in explored_b and prev_b not in frontier_set_b:
                    parent_b[prev_b] = (curr_b, act_b.value)
                    cost_b[prev_b] = cost_b[curr_b] + cost_step_b
                    if prev_b in frontier_set_f or prev_b in explored_f:
                        metrics.solved = True
                        metrics.solution_actions = build_path(prev_b)
                        metrics.total_cost = cost_f.get(prev_b, 0) + cost_b[prev_b]
                        return metrics
                    frontier_b.append(prev_b)
                    frontier_set_b.add(prev_b)

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# 4. Uniform Cost Search (UCS)
# =============================================================================
def uniform_cost_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="UCS")
    start = env.get_initial_state()

    counter = itertools.count()
    frontier = [(0, next(counter), start)]
    best_costs = {start: 0}
    explored = set()
    parent = {start: (None, None)}

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

        curr_cost, _, state = heapq.heappop(frontier)

        if state in explored:
            continue

        if env.is_goal(state):
            metrics.solved = True
            metrics.solution_actions = _reconstruct_path(parent, state)
            metrics.total_cost = curr_cost
            return metrics

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            new_cost = curr_cost + step_cost

            if next_state not in best_costs or new_cost < best_costs[next_state]:
                best_costs[next_state] = new_cost
                parent[next_state] = (state, action.value)
                heapq.heappush(frontier, (new_cost, next(counter), next_state))
                if next_state in explored:
                    explored.remove(next_state)

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# 5. Greedy Best-First Search (GBFS)
# =============================================================================
def greedy_best_first_search(
        env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="GBFS")
    start = env.get_initial_state()

    counter = itertools.count()
    frontier = [(heuristic(start, env), next(counter), start)]
    explored = set()
    parent = {start: (None, None)}
    cost_so_far = {start: 0}

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

        _, _, state = heapq.heappop(frontier)

        if state in explored:
            continue

        if env.is_goal(state):
            metrics.solved = True
            metrics.solution_actions = _reconstruct_path(parent, state)
            metrics.total_cost = cost_so_far.get(state, 0)
            return metrics

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            if next_state not in explored:
                new_cost = cost_so_far.get(state, 0) + step_cost
                if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                    cost_so_far[next_state] = new_cost
                    parent[next_state] = (state, action.value)
                    h_val = heuristic(next_state, env)
                    heapq.heappush(frontier, (h_val, next(counter), next_state))

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# 6. A* Search
# =============================================================================
def a_star_search(
        env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="A*")
    start = env.get_initial_state()

    counter = itertools.count()
    frontier = [(heuristic(start, env), 0, next(counter), start)]
    best_costs = {start: 0}
    explored = set()
    parent = {start: (None, None)}

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

        _, curr_cost, _, state = heapq.heappop(frontier)

        if state in explored:
            continue

        if env.is_goal(state):
            metrics.solved = True
            metrics.solution_actions = _reconstruct_path(parent, state)
            metrics.total_cost = curr_cost
            return metrics

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            new_cost = curr_cost + step_cost

            if next_state not in best_costs or new_cost < best_costs[next_state]:
                best_costs[next_state] = new_cost
                parent[next_state] = (state, action.value)
                f_val = new_cost + heuristic(next_state, env)
                heapq.heappush(frontier, (f_val, new_cost, next(counter), next_state))
                if next_state in explored:
                    explored.remove(next_state)

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# BONUS 1: Weighted A* Search
# =============================================================================
def weighted_a_star_search(
        env: DeepCoreEnv,
        heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
        weight: float = 1.5,
) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="Weighted A*")
    start = env.get_initial_state()

    counter = itertools.count()
    frontier = [(heuristic(start, env) * weight, 0, next(counter), start)]
    best_costs = {start: 0}
    explored = set()
    parent = {start: (None, None)}

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

        _, curr_cost, _, state = heapq.heappop(frontier)

        if state in explored:
            continue

        if env.is_goal(state):
            metrics.solved = True
            metrics.solution_actions = _reconstruct_path(parent, state)
            metrics.total_cost = curr_cost
            return metrics

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            new_cost = curr_cost + step_cost

            if next_state not in best_costs or new_cost < best_costs[next_state]:
                best_costs[next_state] = new_cost
                parent[next_state] = (state, action.value)
                f_val = new_cost + weight * heuristic(next_state, env)
                heapq.heappush(frontier, (f_val, new_cost, next(counter), next_state))
                if next_state in explored:
                    explored.remove(next_state)

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# BONUS 2: Iterative Deepening A* (IDA*)
# =============================================================================
def ida_star_search(
        env: DeepCoreEnv,
        heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
        max_expansions: Optional[int] = None,
) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="IDA*")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics

    h_cache: Dict[DeepCoreState, float] = {}

    def get_h(s: DeepCoreState) -> float:
        if s not in h_cache:
            h_cache[s] = heuristic(s, env)
        return h_cache[s]

    counter = itertools.count()
    threshold = get_h(start)

    while True:
        frontier: List[Tuple[int, int, DeepCoreState, Optional[Tuple[Any, str]]]] = [
            (0, next(counter), start, None)
        ]
        best_in_contour: Dict[DeepCoreState, int] = {start: 0}
        next_threshold = float("inf")

        while frontier:
            if max_expansions is not None and metrics.nodes_expanded >= max_expansions:
                metrics.status_message = "Expansion limit reached."
                return metrics

            metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
            metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(best_in_contour))

            g, _, state, chain = heapq.heappop(frontier)

            if g > best_in_contour.get(state, float("inf")):
                continue

            f = g + get_h(state)
            if f > threshold:
                if f < next_threshold:
                    next_threshold = f
                continue

            if env.is_goal(state):
                metrics.solved = True
                metrics.solution_actions = _unpack_chain(chain)
                metrics.total_cost = g
                return metrics

            metrics.nodes_expanded += 1
            metrics.explored_order.append(state.bot_pos)

            for action, next_state, step_cost in env.get_successors(state):
                metrics.nodes_generated += 1
                new_g = g + step_cost
                f_next = new_g + get_h(next_state)
                if f_next > threshold:
                    if f_next < next_threshold:
                        next_threshold = f_next
                    continue
                if new_g < best_in_contour.get(next_state, float("inf")):
                    best_in_contour[next_state] = new_g
                    heapq.heappush(frontier, (new_g, next(counter), next_state, (chain, action.value)))

        if next_threshold == float("inf"):
            metrics.status_message = "No solution found."
            return metrics

        threshold = next_threshold


# =============================================================================
# BONUS 3: Simplified Memory-Bounded A* (SMA*)
# =============================================================================
def sma_star_search(
        env: DeepCoreEnv,
        heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
        memory_limit: Optional[int] = None,
) -> Optional[SearchMetrics]:
    if memory_limit is None or memory_limit <= 0:
        return a_star_search(env, heuristic)

    metrics = SearchMetrics(algorithm_name="SMA*")
    start = env.get_initial_state()

    counter = itertools.count()
    frontier = [(heuristic(start, env), 0, next(counter), start)]
    best_costs = {start: 0}
    explored = set()
    parent = {start: (None, None)}

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

        if len(frontier) + len(explored) > memory_limit and len(frontier) > 1:
            frontier.sort(key=lambda x: x[0])
            frontier.pop()
            heapq.heapify(frontier)

        _, curr_cost, _, state = heapq.heappop(frontier)

        if state in explored:
            continue

        if env.is_goal(state):
            metrics.solved = True
            metrics.solution_actions = _reconstruct_path(parent, state)
            metrics.total_cost = curr_cost
            return metrics

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            new_cost = curr_cost + step_cost

            if next_state not in best_costs or new_cost < best_costs[next_state]:
                best_costs[next_state] = new_cost
                parent[next_state] = (state, action.value)
                f_val = new_cost + heuristic(next_state, env)
                heapq.heappush(frontier, (f_val, new_cost, next(counter), next_state))
                if next_state in explored:
                    explored.remove(next_state)

    metrics.status_message = "No solution found."
    return metrics
