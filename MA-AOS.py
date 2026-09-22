# ==============================================================================
# MA-AOS: Memetic Algorithm with Adaptive Operator Selection (Probability Matching)
# Controlled baseline for comparison with ISA-MA under identical evaluation budgets.
# ==============================================================================

import pandas as pd
import networkx as nx
import random
import numpy as np
import time
import os
from functools import partial
from multiprocessing import Pool, freeze_support, Manager
import math
import copy
import pickle
from tqdm import tqdm
import collections
from pathlib import Path

# --- Dynamic Path Routing ---
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH_BASE = BASE_DIR / "results"
OUTPUT_PATH_BASE.mkdir(parents=True, exist_ok=True)

# --- Global Configurations ---
POP_SIZE = 30
MAX_GENERATIONS = 1000
ELITISM_RATE = 0.1
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.1
MIN_FREQUENCY = 2
MAX_FREQUENCY = 99
MAX_TRANSFERS = 2
TRANSFER_PENALTY_TIME = 5
ALPHA_UNMET_DEMAND = 10000
BUS_CAPACITY = 50

# --- AOS Parameters (Standard Probability Matching) ---
AOS_ALPHA = 0.1  # Learning rate for reward smoothing
AOS_P_MIN = 0.05  # Minimum probability to prevent operator starvation


class PM_AOS_Tuner:
    """
    Standard Probability Matching Adaptive Operator Selection.
    Tracks credit of operators and updates selection probabilities via roulette wheel.
    """
    def __init__(self, action_names, alpha=AOS_ALPHA, p_min=AOS_P_MIN):
        self.action_names = action_names
        self.num_actions = len(action_names)
        self.alpha = alpha
        self.p_min = p_min
        self.credits = np.zeros(self.num_actions)
        self.probabilities = np.ones(self.num_actions) / self.num_actions

    def select_action(self):
        return np.random.choice(self.num_actions, p=self.probabilities)

    def update(self, action_idx, reward):
        # Update credit using exponential moving average
        self.credits[action_idx] = (1.0 - self.alpha) * self.credits[action_idx] + self.alpha * reward

        # Update probabilities
        total_credit = np.sum(self.credits)
        if total_credit > 1e-8:
            for i in range(self.num_actions):
                self.probabilities[i] = self.p_min + (1.0 - self.num_actions * self.p_min) * (
                            self.credits[i] / total_credit)
        else:
            self.probabilities = np.ones(self.num_actions) / self.num_actions

        # Normalize to prevent floating-point drift
        self.probabilities /= np.sum(self.probabilities)


# --- Data Loading ---
def load_data(base_path):
    base_path_str = str(base_path)
    edge_df = pd.read_csv(os.path.join(base_path_str, 'edge.csv'))
    node_df = pd.read_csv(os.path.join(base_path_str, 'node.csv'))
    od_df = pd.read_csv(os.path.join(base_path_str, 'ODmatrix.csv'))
    stop_nodes_set = set(node_df[node_df['type'] == 'S']['nodeID'])
    G = nx.Graph()
    for _, row in edge_df.iterrows():
        G.add_edge(row['FromNode'], row['ToNode'], weight=row['TT'])

    sp_len_path = os.path.join(base_path_str, 'cached_sp_len.pkl')
    sp_path = os.path.join(base_path_str, 'cached_sp.pkl')
    if os.path.exists(sp_len_path) and os.path.exists(sp_path):
        with open(sp_len_path, 'rb') as f:
            all_pairs_sp_len = pickle.load(f)
        with open(sp_path, 'rb') as f:
            all_pairs_sp = pickle.load(f)
    else:
        od_nodes = set(od_df['OriginID']).union(set(od_df['DestinationID']))
        relevant_nodes = list(od_nodes.union(stop_nodes_set))
        all_pairs_sp_len, all_pairs_sp = {}, {}
        for source in relevant_nodes:
            if source in G:
                lengths, paths = nx.single_source_dijkstra(G, source, weight='weight')
                all_pairs_sp_len[source], all_pairs_sp[source] = lengths, paths
        with open(sp_len_path, 'wb') as f:
            pickle.dump(all_pairs_sp_len, f)
        with open(sp_path, 'wb') as f:
            pickle.dump(all_pairs_sp, f)
    return G, node_df, od_df[od_df['PD'] > 0], all_pairs_sp_len, all_pairs_sp, stop_nodes_set


# --- Core GA & BNDFS Operations ---
def _generate_single_valid_route(sp, stop_nodes_set, min_stops):
    all_stops = list(stop_nodes_set)
    for _ in range(100):
        if len(all_stops) < 2: return []
        origin, dest = random.sample(all_stops, 2)
        new_route = sp.get(origin, {}).get(dest)
        if new_route:
            actual_stops = [node for node in new_route if node in stop_nodes_set]
            if len(actual_stops) >= min_stops:
                return new_route
    return []

def create_heuristic_individual(od_df, sp, sp_len, G, stop_nodes_set, num_routes, min_stops):
    travel_times = od_df.apply(lambda row: sp_len.get(row['OriginID'], {}).get(row['DestinationID'], 0), axis=1)
    rem_od = od_df.copy()
    rem_od['priority'] = rem_od['PD'] * travel_times
    rem_od = rem_od[rem_od['priority'] > 0].sort_values(by='priority', ascending=False)
    routes = []
    for _ in range(num_routes):
        if rem_od.empty: break
        best_od = rem_od.iloc[0]
        nodes = sp.get(best_od['OriginID'], {}).get(best_od['DestinationID'])
        if nodes:
            routes.append(nodes)
            r_set = set(nodes)
            served = rem_od[rem_od['OriginID'].isin(r_set) & rem_od['DestinationID'].isin(r_set)].index
            rem_od = rem_od.drop(served)
    while len(routes) < num_routes:
        r = _generate_single_valid_route(sp, stop_nodes_set, min_stops)
        if r: routes.append(r)
    return {'routes': routes}

def get_route_travel_time(route, sp_len):
    if not route: return float('inf')
    return sum(sp_len.get(route[i], {}).get(route[i + 1], float('inf')) for i in range(len(route) - 1))

def repair_and_validate_individual(ind, G, sp_len, sp, stop_nodes_set, min_stops, max_stops, max_fleet):
    repaired_routes = []
    for r in ind.get('routes', []):
        if not r: continue
        seen = set()
        ordered = [n for n in r if not (n in seen or seen.add(n))]
        if len(ordered) < 2: continue
        clean_r, valid = [ordered[0]], True
        for i in range(len(ordered) - 1):
            seg = sp.get(clean_r[-1], {}).get(ordered[i + 1])
            if not seg: valid = False; break
            clean_r.extend(seg[1:])
        if valid: repaired_routes.append(list(dict.fromkeys(clean_r)))
    ind['routes'] = repaired_routes
    if 'frequencies' in ind:
        for i in range(len(ind['routes'])):
            route = ind['routes'][i]
            stops = [n for n in route if n in stop_nodes_set] if route else []
            if len(stops) < min_stops:
                ind['routes'][i] = _generate_single_valid_route(sp, stop_nodes_set, min_stops)
            elif len(stops) > max_stops:
                while len(stops) > max_stops:
                    if len(stops) > 2:
                        rem = random.choice(stops[1:-1])
                        stops.remove(rem)
                    else:
                        break
                new_r = [stops[0]]
                for k in range(len(stops) - 1):
                    seg = sp.get(new_r[-1], {}).get(stops[k + 1])
                    if seg and len(seg) > 1:
                        new_r.extend(seg[1:])
                    else:
                        new_r.append(stops[k + 1])
                ind['routes'][i] = new_r
        ind['frequencies'] = [max(MIN_FREQUENCY, min(f, MAX_FREQUENCY)) for f in ind['frequencies']]
        while True:
            rtts = [2 * get_route_travel_time(r, sp_len) for r in ind['routes']]
            if any(math.isinf(t) for t in rtts): break
            fleet = [math.ceil((rt * fq) / 60) for rt, fq in zip(rtts, ind['frequencies'])]
            if sum(fleet) <= max_fleet: break
            eff = {i: len([n for n in r if n in stop_nodes_set]) / fleet[i]
                   for i, r in enumerate(ind['routes']) if ind['frequencies'][i] > MIN_FREQUENCY and fleet[i] > 0}
            if not eff: break
            ind['frequencies'][min(eff, key=eff.get)] -= 1
    return ind

def _prepare_network_data_for_individual(ind, sp_len, stop_nodes_set):
    routes, frequencies = ind.get('routes', []), ind.get('frequencies', [])
    if not routes or not frequencies or len(routes) != len(frequencies): return None
    num_routes = len(routes)
    route_sets = [set(r) for r in routes]
    route_node_positions = [{node: pos for pos, node in enumerate(r)} for r in routes]
    route_prefix_times = []
    for r in routes:
        prefix = [0] * len(r)
        for j in range(len(r) - 1):
            prefix[j + 1] = prefix[j] + sp_len.get(r[j], {}).get(r[j + 1], float('inf'))
        route_prefix_times.append(prefix)
    node_to_routes_map = {}
    for i, r_set in enumerate(route_sets):
        for node in r_set:
            if node in stop_nodes_set:
                node_to_routes_map.setdefault(node, []).append(i)
    transfer_hubs, one_hop_routes = {}, {i: set() for i in range(num_routes)}
    for i in range(num_routes):
        for j in range(i + 1, num_routes):
            shared = route_sets[i].intersection(route_sets[j]).intersection(stop_nodes_set)
            if shared:
                transfer_hubs[(i, j)] = transfer_hubs[(j, i)] = shared
                one_hop_routes[i].add(j);
                one_hop_routes[j].add(i)
    wait_times = [30 / f if f > 0 else float('inf') for f in frequencies]
    return {"num_routes": num_routes, "route_sets": route_sets, "route_node_positions": route_node_positions,
            "route_prefix_times": route_prefix_times, "node_to_routes_map": node_to_routes_map,
            "transfer_hubs": transfer_hubs, "one_hop_routes": one_hop_routes, "wait_times": wait_times}

def calculate_performance_metrics(precomputed, od_df):
    if not precomputed:
        total_demand = od_df['PD'].sum()
        return {"fitness": total_demand * ALPHA_UNMET_DEMAND, "unmet_passengers": total_demand,
                "passengers_0_transfer": 0, "total_passenger_time_cost": 0}
    r_pos, r_prefix = precomputed["route_node_positions"], precomputed["route_prefix_times"]
    node_map, one_hop = precomputed["node_to_routes_map"], precomputed["one_hop_routes"]
    hubs, wait_t = precomputed["transfer_hubs"], precomputed["wait_times"]
    total_cost, unmet, p0 = 0, 0, 0
    for row in od_df.itertuples(index=False):
        o, d, dem = row.OriginID, row.DestinationID, row.PD
        if o == d: continue
        best_cost = float('inf')
        transfers = -1
        o_routes, d_routes = node_map.get(o, []), set(node_map.get(d, []))
        for i in set(o_routes).intersection(d_routes):
            p1, p2 = r_pos[i].get(o), r_pos[i].get(d)
            if p1 is not None and p2 is not None:
                cost = abs(r_prefix[i][p2] - r_prefix[i][p1]) + wait_t[i]
                if cost < best_cost: best_cost, transfers = cost, 0
        if MAX_TRANSFERS >= 1 and best_cost == float('inf'):
            for r1 in o_routes:
                for r2 in one_hop[r1].intersection(d_routes):
                    for t_node in hubs.get((r1, r2), []):
                        p1_o, p1_t = r_pos[r1].get(o), r_pos[r1].get(t_node)
                        p2_t, p2_d = r_pos[r2].get(t_node), r_pos[r2].get(d)
                        if all(p is not None for p in [p1_o, p1_t, p2_t, p2_d]):
                            ivt = abs(r_prefix[r1][p1_t] - r_prefix[r1][p1_o]) + abs(
                                r_prefix[r2][p2_d] - r_prefix[r2][p2_t])
                            cost = ivt + wait_t[r1] + wait_t[r2] + TRANSFER_PENALTY_TIME
                            if cost < best_cost: best_cost, transfers = cost, 1
        if transfers != -1:
            total_cost += best_cost * dem
            if transfers == 0: p0 += dem
        else:
            unmet += dem
    return {"fitness": total_cost + unmet * ALPHA_UNMET_DEMAND,
            "total_passenger_time_cost": total_cost,
            "unmet_passengers": unmet,
            "passengers_0_transfer": p0}

def calculate_demand_responsive_frequencies(ind, od_df, sp_len, stop_nodes_set):
    num_routes = len(ind.get('routes', []))
    if num_routes == 0: return []
    temp_ind = copy.deepcopy(ind)
    temp_ind['frequencies'] = [(MIN_FREQUENCY + MAX_FREQUENCY) / 2] * num_routes
    precomputed = _prepare_network_data_for_individual(temp_ind, sp_len, stop_nodes_set)
    if not precomputed: return [MIN_FREQUENCY] * num_routes
    r_pos, r_prefix = precomputed["route_node_positions"], precomputed["route_prefix_times"]
    node_map, one_hop, hubs, wait_t = precomputed["node_to_routes_map"], precomputed["one_hop_routes"], precomputed[
        "transfer_hubs"], precomputed["wait_times"]
    max_loads = [0] * num_routes
    route_loads = {i: collections.defaultdict(int) for i in range(num_routes)}
    for row in od_df.itertuples(index=False):
        o, d, dem = row.OriginID, row.DestinationID, row.PD
        if o == d: continue
        o_routes, d_routes = node_map.get(o, []), set(node_map.get(d, []))
        direct = set(o_routes).intersection(d_routes)
        if direct:
            r_idx = list(direct)[0]
            route_loads[r_idx][(o, d)] += dem
    for r_idx in range(num_routes):
        r = ind['routes'][r_idx]
        if not r: continue
        pos_map = {n: p for p, n in enumerate(r)}
        deltas = collections.defaultdict(int)
        for (u, v), dem in route_loads[r_idx].items():
            pu, pv = pos_map.get(u), pos_map.get(v)
            if pu is not None and pv is not None:
                s, e = min(pu, pv), max(pu, pv)
                deltas[s] += dem;
                deltas[e] -= dem
        curr = 0
        for i in range(len(r) - 1):
            curr += deltas[i]
            max_loads[r_idx] = max(max_loads[r_idx], curr)
    return [max(MIN_FREQUENCY, math.ceil(load / BUS_CAPACITY)) for load in max_loads]

def calculate_fitness_wrapper(ind, sp_len, stop_nodes_set, od_df, G, sp, fitness_cache, total_demand, min_stops,
                              max_stops, max_fleet):
    routes = ind.get('routes', [])
    if not routes: return total_demand * ALPHA_UNMET_DEMAND
    key = ";".join(sorted([",".join(map(str, r)) for r in routes if r]))
    if key in fitness_cache: return fitness_cache[key]
    ind_c = copy.deepcopy(ind)
    ind_c['frequencies'] = calculate_demand_responsive_frequencies(ind_c, od_df, sp_len, stop_nodes_set)
    valid_ind = repair_and_validate_individual(ind_c, G, sp_len, sp, stop_nodes_set, min_stops, max_stops, max_fleet)
    metrics = calculate_performance_metrics(_prepare_network_data_for_individual(valid_ind, sp_len, stop_nodes_set),
                                            od_df)
    fitness_cache[key] = metrics["fitness"]
    return metrics["fitness"]

def selection(pop, fits):
    selected = []
    for _ in range(len(pop)):
        i, j = random.sample(range(len(pop)), 2)
        selected.append(pop[i] if fits[i] < fits[j] else pop[j])
    return selected

def crossover(p1, p2):
    if random.random() > CROSSOVER_RATE or len(p1['routes']) < 2 or len(p2['routes']) < 2:
        return copy.deepcopy(p1), copy.deepcopy(p2)
    pt = random.randint(1, min(len(p1['routes']), len(p2['routes'])) - 1)
    return {'routes': p1['routes'][:pt] + p2['routes'][pt:]}, {'routes': p2['routes'][:pt] + p1['routes'][pt:]}

def mutate(ind, sp, stop_nodes_set, min_stops):
    if random.random() > MUTATION_RATE or not ind['routes']: return ind
    ind_c = copy.deepcopy(ind)
    idx = random.randint(0, len(ind_c['routes']) - 1)
    new_r = _generate_single_valid_route(sp, stop_nodes_set, min_stops)
    if new_r: ind_c['routes'][idx] = new_r
    return ind_c

# --- Local Search Operators ---
def op_intra_route_reroute(ind, **kwargs):
    sp = kwargs['sp']
    ind_c = copy.deepcopy(ind)
    valid = [i for i, r in enumerate(ind_c['routes']) if r and len(r) >= 3]
    if not valid: return ind
    idx = random.choice(valid)
    r = ind_c['routes'][idx]
    s, e = sorted(random.sample(range(len(r)), 2))
    if s + 1 >= e: return ind
    seg = sp.get(r[s], {}).get(r[e], [])
    if seg: ind_c['routes'][idx] = r[:s] + seg + r[e + 1:]
    return ind_c

def op_inter_route_relocate_stop(ind, **kwargs):
    sp, stops, min_s, max_s = kwargs['sp'], kwargs['stop_nodes_set'], kwargs['min_stops'], kwargs['max_stops']
    ind_c = copy.deepcopy(ind)
    routes = ind_c['routes']
    src = [i for i, r in enumerate(routes) if r and len([s for s in r if s in stops]) > min_s]
    dst = [i for i, r in enumerate(routes) if r and len([s for s in r if s in stops]) < max_s]
    if not src or not dst: return ind
    idx_s = random.choice(src)
    valid_d = [i for i in dst if i != idx_s]
    if not valid_d: return ind
    idx_d = random.choice(valid_d)
    r_s, r_d = routes[idx_s], routes[idx_d]
    reloc = [i for i, s in enumerate(r_s) if 0 < i < len(r_s) - 1 and s in stops]
    if not reloc or len(r_d) < 2: return ind
    s_idx = random.choice(reloc)
    node_to_move = r_s[s_idx]
    seg_s = sp.get(r_s[s_idx - 1], {}).get(r_s[s_idx + 1])
    if not seg_s: return ind
    new_s = r_s[:s_idx] + seg_s[1:-1] + r_s[s_idx + 1:]
    ins = random.randint(1, len(r_d) - 1)
    seg_d1 = sp.get(r_d[ins - 1], {}).get(node_to_move)
    seg_d2 = sp.get(node_to_move, {}).get(r_d[ins])
    if not seg_d1 or not seg_d2: return ind
    new_d = r_d[:ins] + seg_d1[1:-1] + [node_to_move] + seg_d2[1:] + r_d[ins:]
    ind_c['routes'][idx_s] = list(dict.fromkeys(new_s))
    ind_c['routes'][idx_d] = list(dict.fromkeys(new_d))
    return ind_c

def op_merge_and_split_routes(ind, **kwargs):
    sp, stops, sp_len, min_s = kwargs['sp'], kwargs['stop_nodes_set'], kwargs['sp_len'], kwargs['min_stops']
    ind_c = copy.deepcopy(ind)
    routes = ind_c['routes']
    if len(routes) < 3: return ind
    best_pair, max_sim = (-1, -1), -1.0
    r_sets = [set(r) for r in routes]
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            u = len(r_sets[i].union(r_sets[j]))
            if u == 0: continue
            sim = len(r_sets[i].intersection(r_sets[j])) / u
            if sim > max_sim: max_sim, best_pair = sim, (i, j)
    if max_sim < 0.1: return ind
    idx1, idx2 = best_pair
    r1, r2 = routes[idx1], routes[idx2]
    conn = sp.get(r1[-1], {}).get(r2[0], [])
    if not conn: return ind
    merged = list(dict.fromkeys(r1 + conn[1:-1] + r2))
    ind_c['routes'][idx1] = merged
    ind_c['routes'][idx2] = []
    longest_idx = np.argmax([len(r) for r in ind_c['routes']])
    if len(ind_c['routes'][longest_idx]) > min_s * 2:
        split_pt = len(ind_c['routes'][longest_idx]) // 2
        r_split = ind_c['routes'][longest_idx]
        ind_c['routes'][longest_idx] = r_split[:split_pt]
        ind_c['routes'][idx2] = r_split[split_pt:]
    return ind_c

def op_worst_routes_regeneration(ind, **kwargs):
    sp, stops, min_s = kwargs['sp'], kwargs['stop_nodes_set'], kwargs['min_stops']
    ind_c = copy.deepcopy(ind)
    if len(ind_c['routes']) > 1:
        target_idx = random.randint(0, len(ind_c['routes']) - 1)
        new_r = _generate_single_valid_route(sp, stops, min_s)
        if new_r: ind_c['routes'][target_idx] = new_r
    return ind_c

# --- Main Optimization Loop for a Single Run of MA-AOS ---
def run_single_ma_aos(G, od_df, sp_len, sp, stop_nodes_set, config, run_seed):
    random.seed(run_seed)
    np.random.seed(run_seed)

    num_routes = config['num_routes']
    min_stops = config['min_stops']
    max_stops = config['max_stops']
    max_fleet = config['max_fleet']

    population = [create_heuristic_individual(od_df, sp, sp_len, G, stop_nodes_set, num_routes, min_stops)
                  for _ in range(POP_SIZE)]

    action_operators = [
        op_intra_route_reroute,
        op_inter_route_relocate_stop,
        op_merge_and_split_routes,
        op_worst_routes_regeneration
    ]
    action_names = ["Reroute", "Relocate", "MergeSplit", "Regenerate"]
    operator_kwargs = {'sp': sp, 'stop_nodes_set': stop_nodes_set, 'sp_len': sp_len,
                       'min_stops': min_stops, 'max_stops': max_stops}

    aos_tuner = PM_AOS_Tuner(action_names)
    fitness_cache = {}
    total_demand = od_df['PD'].sum()

    calc_fit = partial(calculate_fitness_wrapper, sp_len=sp_len, stop_nodes_set=stop_nodes_set,
                       od_df=od_df, G=G, sp=sp, fitness_cache=fitness_cache, total_demand=total_demand,
                       min_stops=min_stops, max_stops=max_stops, max_fleet=max_fleet)

    best_cost = float('inf')
    best_ind = None
    start_time = time.time()

    for gen in range(MAX_GENERATIONS):
        fitnesses = [calc_fit(ind) for ind in population]
        best_idx = np.argmin(fitnesses)
        if fitnesses[best_idx] < best_cost:
            best_cost = fitnesses[best_idx]
            best_ind = copy.deepcopy(population[best_idx])

        sorted_pop = [x for _, x in sorted(zip(fitnesses, population), key=lambda p: p[0])]
        next_pop = sorted_pop[:int(POP_SIZE * ELITISM_RATE)]
        parents = selection(population, fitnesses)

        children = []
        while len(next_pop) + len(children) < POP_SIZE:
            p1, p2 = random.sample(parents, 2)
            c1, c2 = crossover(p1, p2)
            children.append(mutate(c1, sp, stop_nodes_set, min_stops))
            if len(next_pop) + len(children) < POP_SIZE:
                children.append(mutate(c2, sp, stop_nodes_set, min_stops))

        # --- AOS Adaptive Operator Selection and Application ---
        fitnesses_before = [calc_fit(c) for c in children]
        refined_children = []
        for i, child in enumerate(children):
            action_idx = aos_tuner.select_action()
            op = action_operators[action_idx]
            refined = op(copy.deepcopy(child), **operator_kwargs)
            f_after = calc_fit(refined)
            refined_children.append(refined)

            improvement = fitnesses_before[i] - f_after
            reward = max(0.0, np.tanh(improvement / (fitnesses_before[i] + 1e-6)))
            aos_tuner.update(action_idx, reward)

        next_pop.extend(refined_children)
        population = next_pop[:POP_SIZE]

    runtime = time.time() - start_time

    best_ind['frequencies'] = calculate_demand_responsive_frequencies(best_ind, od_df, sp_len, stop_nodes_set)
    valid_best = repair_and_validate_individual(best_ind, G, sp_len, sp, stop_nodes_set, min_stops, max_stops,
                                                max_fleet)
    prep = _prepare_network_data_for_individual(valid_best, sp_len, stop_nodes_set)
    metrics = calculate_performance_metrics(prep, od_df)

    served = total_demand - metrics['unmet_passengers']
    avg_travel_time = (metrics['total_passenger_time_cost'] / served) if served > 0 else float('inf')
    unmet_pct = (metrics['unmet_passengers'] / total_demand) * 100
    direct_pct = (metrics['passengers_0_transfer'] / total_demand) * 100

    return {
        "best_cost": metrics['fitness'],
        "unmet_pct": unmet_pct,
        "direct_pct": direct_pct,
        "avg_travel_time": avg_travel_time,
        "runtime": runtime
    }


# --- 30-Run Batch Evaluation Entry ---
def main():
    NETWORK_CONFIGS = {
        'mandl': {
            "path": BASE_DIR / "data" / "mandl",
            "num_routes": 4, "max_fleet": 99, "max_stops": 15, "min_stops": 3
        },
        'mumford0': {
            "path": BASE_DIR / "data" / "mumford0",
            "num_routes": 12, "max_fleet": 150, "max_stops": 20, "min_stops": 5
        }
    }

    print("=" * 60)
    print("     MA-AOS (Probability Matching Baseline) 30-Run Runner     ")
    print("=" * 60)

    # Run for both networks automatically
    for net_name, config in NETWORK_CONFIGS.items():
        base_path = config["path"]
        
        if not base_path.exists():
            print(f"\n[Warning] Network path {base_path} not found. Skipping {net_name.upper()}...")
            continue

        G, node_df, od_df, sp_len, sp, stop_nodes_set = load_data(base_path)

        num_runs = 30
        run_results = []
        print(f"\nExecuting {num_runs} independent runs for {net_name.upper()}...")

        for seed in tqdm(range(num_runs), desc=f"AOS {net_name.upper()} Progress"):
            res = run_single_ma_aos(G, od_df, sp_len, sp, stop_nodes_set, config, run_seed=seed * 100 + 42)
            run_results.append(res)

        df_runs = pd.DataFrame(run_results)
        out_csv = OUTPUT_PATH_BASE / f"MA_AOS_run_level_results_{net_name}.csv"
        df_runs.to_csv(out_csv, index=False)
        print(f"\nRun-level results saved to: {out_csv}")

        best_cost = df_runs['best_cost'].min()
        mean_cost = df_runs['best_cost'].mean()
        std_cost = df_runs['best_cost'].std()
        mean_unmet = df_runs['unmet_pct'].mean()
        mean_direct = df_runs['direct_pct'].mean()
        mean_time = df_runs['avg_travel_time'].mean()
        mean_runtime = df_runs['runtime'].mean()

        print("\n" + "=" * 60)
        print(f"      Summary Results for Table 6 ({net_name.upper()})      ")
        print("=" * 60)
        print(f"Best Cost:                   {best_cost:.2f}")
        print(f"Mean ± Std:                  {mean_cost:.2f} ± {std_cost:.2f}")
        print(f"Unmet (%):                   {mean_unmet:.2f}%")
        print(f"Direct (%):                  {mean_direct:.2f}%")
        print(f"Avg. Travel Time (min):      {mean_time:.2f}")
        print(f"Algorithmic Runtime (s):     {mean_runtime:.2f}")
        print("=" * 60)

if __name__ == '__main__':
    freeze_support()
    main()
