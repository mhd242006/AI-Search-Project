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
    metrics = SearchMetrics(algorithm_name="Bidirectional")
    start = env.get_initial_state()
    goals = set(env.get_goal_states())

    if start in goals:
        metrics.solved = True
        return metrics

    frontier_f = deque([start])
    frontier_b = deque(goals)
    frontier_set_f = {start}
    frontier_set_b = set(goals)
    explored_f = set()
    explored_b = set()
    parent_f = {start: (None, None)}
    parent_b = {g: (None, None) for g in goals}
    cost_f = {start: 0}
    cost_b = {g: 0 for g in goals}

    def build_path(meet_node):
        path = _reconstruct_path(parent_f, meet_node)
        curr = meet_node
        while curr:
            nxt, act = parent_b.get(curr, (None, None))
            if act is not None:
                path.append(act)
            curr = nxt
        return path

    while frontier_f and frontier_b:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier_f) + len(frontier_b))
        metrics.max_memory_size = max(metrics.max_memory_size,
                                      len(frontier_f) + len(frontier_b) + len(explored_f) + len(explored_b))

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
                if nxt_f in frontier_set_b or nxt_f in explored_b:
                    metrics.solved = True
                    metrics.solution_actions = build_path(nxt_f)
                    metrics.total_cost = cost_f[nxt_f] + cost_b.get(nxt_f, 0)
                    return metrics
                frontier_f.append(nxt_f)
                frontier_set_f.add(nxt_f)

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
# 4. Uniform Cost Search (UCS) — [STUDENT TODO]
# =============================================================================
def uniform_cost_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="UCS")
    start = env.get_initial_state()

    frontier = [(0, id(start), start)]
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

            if next_state not in explored:
                if new_cost < best_costs.get(next_state, float('inf')):
                    best_costs[next_state] = new_cost
                    parent[next_state] = (state, action.value)
                    heapq.heappush(frontier, (new_cost, id(next_state), next_state))

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# 5. Greedy Best-First Search (GBFS) — [STUDENT TODO]
# =============================================================================
def greedy_best_first_search(
        env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    """TODO: Implement Greedy Best-First Search ordering the priority queue strictly by h(n)."""
    raise NotImplementedError


# =============================================================================
# 6. A* Search — [STUDENT TODO]
# =============================================================================
def a_star_search(
        env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    """TODO: Implement A* Graph Search ordering the priority queue by f(n) = g(n) + h(n)."""
    raise NotImplementedError


# =============================================================================
# BONUS 1: Weighted A* Search — [STUDENT BONUS TODO]
# =============================================================================
def weighted_a_star_search(
        env: DeepCoreEnv,
        heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
        weight: float = 1.5,
) -> Optional[SearchMetrics]:
    """TODO (Bonus): Implement Weighted A* ordering the frontier by f(n) = g(n) + W * h(n)."""
    raise NotImplementedError


# =============================================================================
# BONUS 2: Iterative Deepening A* (IDA*) — [STUDENT BONUS TODO]
# =============================================================================
def ida_star_search(
        env: DeepCoreEnv,
        heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
        max_expansions: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """TODO (Bonus): Implement Iterative Deepening A* (IDA*) using f-cost threshold contours."""
    raise NotImplementedError


# =============================================================================
# BONUS 3: Simplified Memory-Bounded A* (SMA*) — [STUDENT BONUS TODO]
# =============================================================================
def sma_star_search(
        env: DeepCoreEnv,
        heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
        memory_limit: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """TODO (Bonus): Implement Simplified Memory-Bounded A* (SMA*)."""
    raise NotImplementedError
