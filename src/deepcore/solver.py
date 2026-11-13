"""
solver.py
Student implementation file for DeepCore search algorithms.

- Breadth-First Search (BFS) is provided below as a complete reference example
  showing how to interact with `env.get_successors(state)` and `SearchMetrics`.
- Implement the remaining required algorithms. Do not change the public
  function signatures or the environment/state modules.
"""

from collections import deque
import heapq
import math
from typing import Callable, Dict, List, Optional, Set, Tuple

from .environment import DeepCoreEnv
from .state import DeepCoreState, SearchMetrics


def _reconstruct_path(
        parent_map: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]],
        goal_state: DeepCoreState,
) -> List[str]:
    """Helper function to trace actions from the initial state to `goal_state`."""
    actions: List[str] = []
    curr: Optional[DeepCoreState] = goal_state
    while curr is not None:
        prev, act = parent_map.get(curr, (None, None))
        if act is not None:
            actions.append(act)
        curr = prev
    actions.reverse()
    return actions


# =============================================================================
# 1. Breadth-First Search (BFS) — [PROVIDED AS REFERENCE EXAMPLE]
# =============================================================================
def breadth_first_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    """
    Reference Graph-Search BFS implementation (FIFO queue).
    Minimizes the number of steps, but ignores terrain/payload energy costs!
    """
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
# 2. Depth-First Search (DFS) & Iterative Deepening Search (IDS) — [STUDENT TODO]
# =============================================================================
def depth_first_search(env: DeepCoreEnv, max_expansions: Optional[int] = None) -> Optional[SearchMetrics]:
    """TODO: Implement Depth-First Graph Search using a LIFO Stack."""
    raise NotImplementedError


def iterative_deepening_search(
        env: DeepCoreEnv,
        max_depth: Optional[int] = None,
        max_expansions: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """TODO: Implement Iterative Deepening Search (IDS)."""
    raise NotImplementedError


# =============================================================================
# 3. Bidirectional Search — [STUDENT TODO]
# =============================================================================
def bidirectional_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    """
    TODO: Implement Bidirectional Search expanding forward from `env.get_initial_state()`
    via `env.get_successors(s)` and backward from `env.get_goal_states()` via `env.get_predecessors(s)`.
    """
    raise NotImplementedError


# =============================================================================
# 4. Uniform Cost Search (UCS) — [STUDENT TODO]
# =============================================================================
def uniform_cost_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    """TODO: Implement Uniform Cost Search ordering the priority queue by path cost g(n)."""
    raise NotImplementedError


# =============================================================================
# 5. Greedy Best-First Search (GBFS) — [STUDENT TODO]
# =============================================================================
def greedy_best_first_search(
        env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="GBFS")
    start = env.get_initial_state()

    if env.is_goal(start):
        metrics.solved = True
        return metrics

    frontier = [(heuristic(start, env), id(start), start)]
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
            metrics.total_cost = cost_so_far[state]
            return metrics

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            if next_state not in explored:
                new_cost = cost_so_far[state] + step_cost
                if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                    cost_so_far[next_state] = new_cost
                    parent[next_state] = (state, action.value)
                    h_val = heuristic(next_state, env)
                    heapq.heappush(frontier, (h_val, id(next_state), next_state))

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# 6. A* Search — [STUDENT TODO]
# =============================================================================
def a_star_search(
        env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="A*")
    start = env.get_initial_state()

    frontier = [(heuristic(start, env), 0, id(start), start)]
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

            if next_state not in explored:
                if new_cost < best_costs.get(next_state, float('inf')):
                    best_costs[next_state] = new_cost
                    parent[next_state] = (state, action.value)
                    f_val = new_cost + heuristic(next_state, env)
                    heapq.heappush(frontier, (f_val, new_cost, id(next_state), next_state))

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# BONUS 1: Weighted A* Search — [STUDENT BONUS TODO]
# =============================================================================
def weighted_a_star_search(
        env: DeepCoreEnv,
        heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
        weight: float = 1.5,
) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="Weighted A*")
    start = env.get_initial_state()

    frontier = [(heuristic(start, env) * weight, 0, id(start), start)]
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

            if next_state not in explored:
                if new_cost < best_costs.get(next_state, float('inf')):
                    best_costs[next_state] = new_cost
                    parent[next_state] = (state, action.value)
                    f_val = new_cost + weight * heuristic(next_state, env)
                    heapq.heappush(frontier, (f_val, new_cost, id(next_state), next_state))

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# BONUS 2: Iterative Deepening A* (IDA*) — [STUDENT BONUS TODO]
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

    threshold = heuristic(start, env)

    while True:
        stack = [(start, 0, [start])]
        parent = {start: (None, None)}
        next_threshold = float('inf')

        while stack:
            if max_expansions is not None and metrics.nodes_expanded >= max_expansions:
                metrics.status_message = "Expansion limit reached."
                return metrics

            metrics.max_frontier_size = max(metrics.max_frontier_size, len(stack))
            metrics.max_memory_size = max(metrics.max_memory_size, len(stack))

            state, g, path = stack.pop()

            f = g + heuristic(state, env)
            if f > threshold:
                if f < next_threshold:
                    next_threshold = f
                continue

            metrics.nodes_expanded += 1
            metrics.explored_order.append(state.bot_pos)

            if env.is_goal(state):
                metrics.solved = True
                metrics.solution_actions = _reconstruct_path(parent, state)
                metrics.total_cost = g
                return metrics

            for action, next_state, step_cost in env.get_successors(state):
                metrics.nodes_generated += 1
                if next_state not in path:
                    parent[next_state] = (state, action.value)
                    stack.append((next_state, g + step_cost, path + [next_state]))

        if next_threshold == float('inf'):
            metrics.status_message = "No solution found."
            return metrics

        threshold = next_threshold


# =============================================================================
# BONUS 3: Simplified Memory-Bounded A* (SMA*) — [STUDENT BONUS TODO]
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

    frontier = [(heuristic(start, env), 0, id(start), start)]
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

            if next_state not in explored:
                if new_cost < best_costs.get(next_state, float('inf')):
                    best_costs[next_state] = new_cost
                    parent[next_state] = (state, action.value)
                    f_val = new_cost + heuristic(next_state, env)
                    heapq.heappush(frontier, (f_val, new_cost, id(next_state), next_state))

    metrics.status_message = "No solution found."
    return metrics
