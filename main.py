"""
ISA-MA: Interpretable State-Aware Memetic Algorithm.
This module implements the transit network optimization framework 
as described in the manuscript.
"""
import pandas as pd
import networkx as nx
import random
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# --- 1. Configuration Parameters (These will be overridden by the config dictionary in the main function) ---
BASE_PATH = ""
OUTPUT_PATH_BASE = BASE_DIR / "output"
POP_SIZE = 30
MAX_GENERATIONS = 1000
ELITISM_RATE = 0.1
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.1
NUM_ROUTES = 4
MAX_STOPS_PER_ROUTE = 15
MIN_STOPS_PER_ROUTE = 3
MIN_FREQUENCY = 2
MAX_FREQUENCY = 99
MAX_TOTAL_FLEET = 99
MAX_TRANSFERS = 2
TRANSFER_PENALTY_TIME = 5
ALPHA_UNMET_DEMAND = 10000
BUS_CAPACITY = 50

# --- RL Tuner Parameters ---
RL_LEARNING_RATE = 0.2
RL_DISCOUNT_FACTOR = 0.9
RL_EPSILON_START = 1.0
RL_EPSILON_END = 0.01
RL_EPSILON_DECAY = 0.985
RL_STAGNATION_THRESHOLD = 0.01


# --- 2. Reinforcement Learning Core (Advanced_RL_Tuner from text.txt) ---
class Advanced_RL_Tuner:
    def __init__(self, actions, stagnation_threshold=0.001, temperature=0.1):
        self.temperature = temperature
        self.actions = actions
        self.num_actions = len(actions)
        self.states = [0, 1, 2, 3]
        self.q_table = np.zeros((len(self.states), self.num_actions))
        self.epsilon = RL_EPSILON_START
        self.convergence_history = collections.deque(maxlen=10)
        self.diversity_threshold_ratio = 0.1
        self.stagnation_threshold = stagnation_threshold

    def get_state(self, best_fitness_current_gen, all_fitnesses_current_gen):
        is_stagnant = False
        if len(self.convergence_history) == self.convergence_history.maxlen:
            mean_fit = np.mean(list(self.convergence_history))
            std_fit = np.std(list(self.convergence_history))
            if mean_fit != 0 and (std_fit / abs(mean_fit)) < self.stagnation_threshold:
                is_stagnant = True
        self.convergence_history.append(best_fitness_current_gen)
        fitness_std = np.std(all_fitnesses_current_gen)
        fitness_mean = np.mean(all_fitnesses_current_gen)
        is_low_diversity = (fitness_std / abs(
            fitness_mean)) < self.diversity_threshold_ratio if fitness_mean != 0 else True
        if is_stagnant and not is_low_diversity: return 0
        if is_stagnant and is_low_diversity: return 1
        if not is_stagnant and not is_low_diversity: return 2
        if not is_stagnant and is_low_diversity: return 3
        return 2

    def choose_action_probabilities(self, state):
        q_values = self.q_table[state, :]
        if self.temperature > 1e-6:
            q_values = q_values / self.temperature
        q_values_norm = q_values - np.max(q_values)
        exp_q = np.exp(q_values_norm)
        probabilities = exp_q / np.sum(exp_q)
        if not np.isclose(np.sum(probabilities), 1.0):
            probabilities = np.ones(self.num_actions) / self.num_actions
        return probabilities

    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        future_optimal_value = np.max(self.q_table[next_state, :])
        new_value = old_value + RL_LEARNING_RATE * (reward + RL_DISCOUNT_FACTOR * future_optimal_value - old_value)
        self.q_table[state, action] = new_value

    def decay_epsilon(self):
        if self.epsilon > RL_EPSILON_END: self.epsilon *= RL_EPSILON_DECAY


# --- 3. Data Loading and Preprocessing ---
def load_data(base_path):
    print(f"Loading data from '{base_path}'...")
    try:
        edge_df = pd.read_csv(os.path.join(base_path, 'edge.csv'))
        node_df = pd.read_csv(os.path.join(base_path, 'node.csv'))
        od_df = pd.read_csv(os.path.join(base_path, 'ODmatrix.csv'))
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None, None, None, None, None, None
    if 'type' not in node_df.columns:
        print("Error: 'type' field is missing in node.csv. Please ensure this field exists.")
        return None, None, None, None, None, None
    stop_nodes_set = set(node_df[node_df['type'] == 'S']['nodeID'])
    print(f"Identified {len(stop_nodes_set)} actual bus stops.")
    G = nx.Graph()
    for _, row in edge_df.iterrows():
        G.add_edge(row['FromNode'], row['ToNode'], weight=row['TT'])
    sp_len_path, sp_path = os.path.join(base_path, 'cached_sp_len.pkl'), os.path.join(base_path, 'cached_sp.pkl')
    if os.path.exists(sp_len_path) and os.path.exists(sp_path):
        print("Cached shortest path files found, loading...")
        with open(sp_len_path, 'rb') as f:
            all_pairs_sp_len = pickle.load(f)
        with open(sp_path, 'rb') as f:
            all_pairs_sp = pickle.load(f)
    else:
        od_nodes = set(od_df['OriginID']).union(set(od_df['DestinationID']))
        relevant_nodes = list(od_nodes.union(stop_nodes_set))
        all_pairs_sp_len, all_pairs_sp = {}, {}
        for source in tqdm(relevant_nodes, desc="Calculating Shortest Paths"):
            if source in G:
                lengths, paths = nx.single_source_dijkstra(G, source, weight='weight')
                all_pairs_sp_len[source], all_pairs_sp[source] = lengths, paths
        print("\nShortest path calculation complete. Caching results...")
        with open(sp_len_path, 'wb') as f:
            pickle.dump(all_pairs_sp_len, f)
        with open(sp_path, 'wb') as f:
            pickle.dump(all_pairs_sp, f)
    print("Data loading and preprocessing complete.")
    return G, node_df, od_df[od_df['PD'] > 0], all_pairs_sp_len, all_pairs_sp, stop_nodes_set


# --- 4. Core Functions (including GA operators) ---
def _generate_single_valid_route(sp, stop_nodes_set):
    all_stops = list(stop_nodes_set)
    attempts = 0
    while attempts < 100:
        if len(all_stops) < 2: return []
        origin, dest = random.sample(all_stops, 2)
        new_route = sp.get(origin, {}).get(dest)
        if new_route:
            actual_stops_in_route = [node for node in new_route if node in stop_nodes_set]
            if len(actual_stops_in_route) >= MIN_STOPS_PER_ROUTE:
                return new_route
        attempts += 1
    return []


def create_heuristic_individual(od_df, sp, sp_len, G, stop_nodes_set):
    travel_times = od_df.apply(lambda row: sp_len.get(row['OriginID'], {}).get(row['DestinationID'], 0), axis=1)
    remaining_od = od_df.copy()
    remaining_od['priority'] = remaining_od['PD'] * travel_times
    remaining_od = remaining_od[remaining_od['priority'] > 0].sort_values(by='priority', ascending=False)
    routes = []
    for _ in range(NUM_ROUTES):
        if remaining_od.empty: break
        best_od_series = remaining_od.iloc[0]
        origin, dest = best_od_series['OriginID'], best_od_series['DestinationID']
        new_route_nodes = sp.get(origin, {}).get(dest)
        if new_route_nodes:
            routes.append(new_route_nodes)
            new_route_set = set(new_route_nodes)
            served_indices = remaining_od[
                remaining_od['OriginID'].isin(new_route_set) & remaining_od['DestinationID'].isin(new_route_set)].index
            remaining_od = remaining_od.drop(served_indices)
    while len(routes) < NUM_ROUTES:
        new_route = _generate_single_valid_route(sp, stop_nodes_set)
        if new_route: routes.append(new_route)
    return {'routes': routes}


def load_initial_solution(file_path):
    if file_path is None or not os.path.exists(file_path): return None
    print(f"Loading initial solution from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        routes = [[int(n.strip("'\" ")) for n in str(row['node_seq']).split(',')] for _, row in df.iterrows()]
        initial_individual = {'routes': routes}
        print("Initial solution loaded successfully, frequencies will be calculated dynamically.")
        return initial_individual
    except Exception as e:
        print(f"Warning: Error loading or parsing initial solution: {e}.")
        return None


def get_route_travel_time(route, sp_len):
    travel_time = 0
    if not route: return float('inf')
    for i in range(len(route) - 1):
        travel_time += sp_len.get(route[i], {}).get(route[i + 1], float('inf'))
    return travel_time


def repair_and_validate_individual(individual, G, sp_len, sp, stop_nodes_set):
    repaired_routes = []
    for original_route in individual.get('routes', []):
        if not original_route:
            repaired_routes.append([])
            continue
        seen = set()
        ordered_unique = [n for n in original_route if not (n in seen or seen.add(n))]
        if len(ordered_unique) < 2:
            repaired_routes.append([])
            continue
        clean_route, path_valid = [ordered_unique[0]], True
        for i in range(len(ordered_unique) - 1):
            u, v = clean_route[-1], ordered_unique[i + 1]
            if u == v: continue
            path_segment = sp.get(u, {}).get(v)
            if not path_segment:
                path_valid = False
                break
            clean_route.extend(path_segment[1:])
        if path_valid:
            repaired_routes.append(list(dict.fromkeys(clean_route)))
        else:
            repaired_routes.append([])
    individual['routes'] = repaired_routes
    if 'frequencies' in individual:
        for i in range(len(individual['routes'])):
            route = individual['routes'][i]
            actual_stops_in_route = [node for node in route if node in stop_nodes_set] if route else []
            if len(actual_stops_in_route) < MIN_STOPS_PER_ROUTE:
                individual['routes'][i] = _generate_single_valid_route(sp, stop_nodes_set)
                route = individual['routes'][i]
                actual_stops_in_route = [node for node in route if node in stop_nodes_set] if route else []
            if len(actual_stops_in_route) > MAX_STOPS_PER_ROUTE:
                while len(actual_stops_in_route) > MAX_STOPS_PER_ROUTE:
                    if len(actual_stops_in_route) > 2:
                        stop_to_remove = random.choice(actual_stops_in_route[1:-1])
                        actual_stops_in_route.remove(stop_to_remove)
                    else:
                        break
                if len(actual_stops_in_route) >= 2:
                    new_repaired_route = [actual_stops_in_route[0]]
                    for k in range(len(actual_stops_in_route) - 1):
                        u, v = new_repaired_route[-1], actual_stops_in_route[k + 1]
                        path_segment = sp.get(u, {}).get(v)
                        if path_segment and len(path_segment) > 1:
                            new_repaired_route.extend(path_segment[1:])
                        else:
                            new_repaired_route.append(v)
                    individual['routes'][i] = new_repaired_route
                else:
                    individual['routes'][i] = _generate_single_valid_route(sp, stop_nodes_set)
        individual['frequencies'] = [max(MIN_FREQUENCY, min(f, MAX_FREQUENCY)) for f in individual['frequencies']]
        while True:
            round_trip_times = [2 * get_route_travel_time(r, sp_len) for r in individual['routes']]
            if any(math.isinf(t) for t in round_trip_times): break
            fleet_per_route = [math.ceil((rt * fq) / 60) for rt, fq in zip(round_trip_times, individual['frequencies'])]
            if sum(fleet_per_route) <= MAX_TOTAL_FLEET: break
            efficiencies = {i: len([node for node in r if node in stop_nodes_set]) / fleet_per_route[i] for i, r in
                            enumerate(individual['routes']) if
                            individual['frequencies'][i] > MIN_FREQUENCY and fleet_per_route[i] > 0}
            if not efficiencies: break
            route_idx_to_reduce = min(efficiencies, key=efficiencies.get)
            individual['frequencies'][route_idx_to_reduce] -= 1
    return individual


def _calculate_best_two_transfer_path(origin, dest, r1, r_mid, r3, transfer_hubs, route_prefix_times,
                                      route_node_positions):
    min_ivt1, best_t1_node = float('inf'), None
    t1_nodes = transfer_hubs.get((r1, r_mid), [])
    if not t1_nodes: return float('inf')
    origin_pos_r1 = route_node_positions[r1].get(origin)
    if origin_pos_r1 is None: return float('inf')
    for t1 in t1_nodes:
        t1_pos_r1 = route_node_positions[r1].get(t1)
        if t1_pos_r1 is not None:
            ivt = abs(route_prefix_times[r1][t1_pos_r1] - route_prefix_times[r1][origin_pos_r1])
            if ivt < min_ivt1: min_ivt1, best_t1_node = ivt, t1
    if best_t1_node is None: return float('inf')
    min_ivt3, best_t2_node = float('inf'), None
    t2_nodes = transfer_hubs.get((r_mid, r3), [])
    if not t2_nodes: return float('inf')
    dest_pos_r3 = route_node_positions[r3].get(dest)
    if dest_pos_r3 is None: return float('inf')
    for t2 in t2_nodes:
        t2_pos_r3 = route_node_positions[r3].get(t2)
        if t2_pos_r3 is not None:
            ivt = abs(route_prefix_times[r3][dest_pos_r3] - route_prefix_times[r3][t2_pos_r3])
            if ivt < min_ivt3: min_ivt3, best_t2_node = ivt, t2
    if best_t2_node is None: return float('inf')
    t1_pos_rmid, t2_pos_rmid = route_node_positions[r_mid].get(best_t1_node), route_node_positions[r_mid].get(
        best_t2_node)
    if t1_pos_rmid is None or t2_pos_rmid is None: return float('inf')
    min_ivt2 = abs(route_prefix_times[r_mid][t2_pos_rmid] - route_prefix_times[r_mid][t1_pos_rmid])
    return min_ivt1 + min_ivt2 + min_ivt3


def _prepare_network_data_for_individual(individual, sp_len, stop_nodes_set):
    routes, frequencies = individual.get('routes', []), individual.get('frequencies', [])
    if not routes or not frequencies or len(routes) != len(frequencies): return None
    num_routes = len(routes)
    route_sets = [set(r) for r in routes]
    route_node_positions = [{node: pos for pos, node in enumerate(r)} for r in routes]
    route_prefix_times = []
    for r in routes:
        prefix_times = [0] * len(r)
        for j in range(len(r) - 1):
            prefix_times[j + 1] = prefix_times[j] + sp_len.get(r[j], {}).get(r[j + 1], float('inf'))
        route_prefix_times.append(prefix_times)
    node_to_routes_map = {}
    for i, r_set in enumerate(route_sets):
        for node in r_set:
            if node in stop_nodes_set:
                node_to_routes_map.setdefault(node, []).append(i)
    transfer_hubs, one_hop_routes = {}, {i: set() for i in range(num_routes)}
    for i in range(num_routes):
        for j in range(i + 1, num_routes):
            shared_stops = route_sets[i].intersection(route_sets[j]).intersection(stop_nodes_set)
            if shared_stops:
                transfer_hubs[(i, j)] = transfer_hubs[(j, i)] = shared_stops
                one_hop_routes[i].add(j)
                one_hop_routes[j].add(i)
    wait_times = [30 / f if f > 0 else float('inf') for f in frequencies]
    return {"num_routes": num_routes, "route_sets": route_sets, "route_node_positions": route_node_positions,
            "route_prefix_times": route_prefix_times, "node_to_routes_map": node_to_routes_map,
            "transfer_hubs": transfer_hubs, "one_hop_routes": one_hop_routes, "wait_times": wait_times}


def calculate_performance_metrics(precomputed_data, od_df):
    if not precomputed_data:
        total_demand = od_df['PD'].sum()
        return {"fitness": total_demand * ALPHA_UNMET_DEMAND, "unmet_passengers": total_demand,
                "passengers_0_transfer": 0,
                "total_passenger_time_cost": 0, "passengers_1_transfer": 0, "passengers_2_transfer": 0,
                "total_in_vehicle_time": 0, "total_wait_time": 0, "total_transfer_time": 0}
    route_node_positions, route_prefix_times = precomputed_data["route_node_positions"], precomputed_data[
        "route_prefix_times"]
    node_to_routes_map, one_hop_routes = precomputed_data["node_to_routes_map"], precomputed_data["one_hop_routes"]
    transfer_hubs, wait_times = precomputed_data["transfer_hubs"], precomputed_data["wait_times"]
    total_passenger_time_cost, unmet_demand, total_wait_t, total_transfer_t, total_in_vehicle_t, p0, p1, p2 = 0, 0, 0, 0, 0, 0, 0, 0
    for od_row in od_df.itertuples(index=False):
        origin, dest, demand = od_row.OriginID, od_row.DestinationID, od_row.PD
        if origin == dest: continue
        best_path = {'cost': float('inf'), 'transfers': -1, 'ivt': 0, 'wait': 0, 'transfer_p': 0}
        origin_routes, dest_routes_set = node_to_routes_map.get(origin, []), set(node_to_routes_map.get(dest, []))
        for i in set(origin_routes).intersection(dest_routes_set):
            pos1, pos2 = route_node_positions[i].get(origin), route_node_positions[i].get(dest)
            if pos1 is not None and pos2 is not None:
                ivt = abs(route_prefix_times[i][pos2] - route_prefix_times[i][pos1])
                cost = ivt + wait_times[i]
                if cost < best_path['cost']: best_path.update(
                    {'cost': cost, 'transfers': 0, 'ivt': ivt, 'wait': wait_times[i], 'transfer_p': 0})
        if MAX_TRANSFERS >= 1:
            for r1 in origin_routes:
                for r2 in one_hop_routes[r1].intersection(dest_routes_set):
                    min_ivt = float('inf')
                    for t_node in transfer_hubs.get((r1, r2), []):
                        pos1_o, pos1_t = route_node_positions[r1].get(origin), route_node_positions[r1].get(t_node)
                        pos2_t, pos2_d = route_node_positions[r2].get(t_node), route_node_positions[r2].get(dest)
                        if all(p is not None for p in [pos1_o, pos1_t, pos2_t, pos2_d]):
                            min_ivt = min(min_ivt,
                                          abs(route_prefix_times[r1][pos1_t] - route_prefix_times[r1][pos1_o]) + abs(
                                              route_prefix_times[r2][pos2_d] - route_prefix_times[r2][pos2_t]))
                    if not math.isinf(min_ivt):
                        cost = min_ivt + wait_times[r1] + wait_times[r2] + TRANSFER_PENALTY_TIME
                        if cost < best_path['cost']: best_path.update(
                            {'cost': cost, 'transfers': 1, 'ivt': min_ivt, 'wait': wait_times[r1] + wait_times[r2],
                             'transfer_p': TRANSFER_PENALTY_TIME})
        if MAX_TRANSFERS >= 2 and best_path['transfers'] != 0:
            for r1 in origin_routes:
                for r_mid in one_hop_routes[r1]:
                    for r3 in one_hop_routes[r_mid].intersection(dest_routes_set):
                        if r3 == r1 or r3 == r_mid: continue
                        total_ivt = _calculate_best_two_transfer_path(origin, dest, r1, r_mid, r3, transfer_hubs,
                                                                      route_prefix_times, route_node_positions)
                        if not math.isinf(total_ivt):
                            wait = wait_times[r1] + wait_times[r_mid] + wait_times[r3]
                            cost = total_ivt + wait + 2 * TRANSFER_PENALTY_TIME
                            if cost < best_path['cost']: best_path.update(
                                {'cost': cost, 'transfers': 2, 'ivt': total_ivt, 'wait': wait,
                                 'transfer_p': 2 * TRANSFER_PENALTY_TIME})
        if best_path['transfers'] != -1:
            total_passenger_time_cost += best_path['cost'] * demand
            total_wait_t += best_path['wait'] * demand
            total_transfer_t += best_path['transfer_p'] * demand
            total_in_vehicle_t += best_path['ivt'] * demand
            if best_path['transfers'] == 0:
                p0 += demand
            elif best_path['transfers'] == 1:
                p1 += demand
            else:
                p2 += demand
        else:
            unmet_demand += demand
    fitness = total_passenger_time_cost + unmet_demand * ALPHA_UNMET_DEMAND
    return {"fitness": fitness, "total_passenger_time_cost": total_passenger_time_cost, "total_wait_time": total_wait_t,
            "total_transfer_time": total_transfer_t, "total_in_vehicle_time": total_in_vehicle_t,
            "unmet_passengers": unmet_demand, "passengers_0_transfer": p0, "passengers_1_transfer": p1,
            "passengers_2_transfer": p2}


def calculate_demand_responsive_frequencies(individual, od_df, sp_len, stop_nodes_set):
    num_routes = len(individual.get('routes', []))
    if num_routes == 0: return []
    avg_freq = (MIN_FREQUENCY + MAX_FREQUENCY) / 2
    temp_individual = copy.deepcopy(individual)
    temp_individual['frequencies'] = [avg_freq] * num_routes
    precomputed_data = _prepare_network_data_for_individual(temp_individual, sp_len, stop_nodes_set)
    if not precomputed_data: return [MIN_FREQUENCY] * num_routes
    route_segment_loads, _, _ = _calculate_passenger_loads_on_routes(temp_individual, precomputed_data, od_df,
                                                                     stop_nodes_set)
    max_loads = [0] * num_routes
    for r_idx in range(num_routes):
        route = individual['routes'][r_idx]
        if not route: continue
        pos_map = {node: pos for pos, node in enumerate(route)}
        load_deltas = collections.defaultdict(int)
        for (o, d), demand in route_segment_loads[r_idx].items():
            pos_o, pos_d = pos_map.get(o), pos_map.get(d)
            if pos_o is not None and pos_d is not None:
                start, end = min(pos_o, pos_d), max(pos_o, pos_d)
                if start < end:
                    load_deltas[start] += demand
                    load_deltas[end] -= demand
        current_load = 0
        for i in range(len(route) - 1):
            current_load += load_deltas.get(i, 0)
            max_loads[r_idx] = max(max_loads[r_idx], current_load)
    frequencies = [math.ceil(load / BUS_CAPACITY) if BUS_CAPACITY > 0 else MAX_FREQUENCY for load in max_loads]
    return frequencies


def calculate_fitness_wrapper(individual, sp_len, stop_nodes_set, od_df, G, sp, fitness_cache, total_demand):
    routes = individual.get('routes', [])
    if not routes:
        return total_demand * ALPHA_UNMET_DEMAND
    canonical_routes = sorted([",".join(map(str, r)) for r in routes if r])
    cache_key = ";".join(canonical_routes)
    if cache_key in fitness_cache:
        return fitness_cache[cache_key]
    ind_copy = copy.deepcopy(individual)
    demand_frequencies = calculate_demand_responsive_frequencies(ind_copy, od_df, sp_len, stop_nodes_set)
    ind_copy['frequencies'] = demand_frequencies
    valid_individual = repair_and_validate_individual(ind_copy, G, sp_len, sp, stop_nodes_set)
    precomputed_data = _prepare_network_data_for_individual(valid_individual, sp_len, stop_nodes_set)
    metrics = calculate_performance_metrics(precomputed_data, od_df)
    fitness = metrics["fitness"]
    fitness_cache[cache_key] = fitness
    return fitness


def selection(population, fitnesses):
    selected = []
    for _ in range(len(population)):
        i, j = random.sample(range(len(population)), 2)
        selected.append(population[i] if fitnesses[i] < fitnesses[j] else population[j])
    return selected


def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)
    p1_routes, p2_routes = parent1['routes'], parent2['routes']
    if len(p1_routes) < 2 or len(p2_routes) < 2:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)
    min_len = min(len(p1_routes), len(p2_routes))
    if min_len < 2: return copy.deepcopy(parent1), copy.deepcopy(parent2)
    crossover_point = random.randint(1, min_len - 1)
    c1_routes = p1_routes[:crossover_point] + p2_routes[crossover_point:]
    c2_routes = p2_routes[:crossover_point] + p1_routes[crossover_point:]
    return {'routes': c1_routes}, {'routes': c2_routes}


def mutate(individual, G, sp, stop_nodes_set):
    if random.random() > MUTATION_RATE or not individual['routes']:
        return individual
    ind_copy = copy.deepcopy(individual)
    route_idx_to_mutate = random.randint(0, len(ind_copy['routes']) - 1)
    new_route = _generate_single_valid_route(sp, stop_nodes_set)
    if new_route:
        ind_copy['routes'][route_idx_to_mutate] = new_route
    return ind_copy


# --- 5. Local Search Operators ---

# --- 5.1 Helper Functions (for advanced operators) ---
def _calculate_passenger_loads_on_routes(individual, precomputed_data, od_df, stop_nodes_set):
    num_routes = precomputed_data['num_routes']
    route_node_positions, route_prefix_times = precomputed_data["route_node_positions"], precomputed_data[
        "route_prefix_times"]
    node_to_routes_map, one_hop_routes = precomputed_data["node_to_routes_map"], precomputed_data["one_hop_routes"]
    transfer_hubs, wait_times = precomputed_data["transfer_hubs"], precomputed_data["wait_times"]
    route_segment_loads = {i: collections.defaultdict(int) for i in range(num_routes)}
    unmet_od_pairs, served_od_info = [], []

    for od_row in od_df.itertuples(index=False):
        origin, dest, demand = od_row.OriginID, od_row.DestinationID, od_row.PD
        if origin == dest: continue
        best_path = {'cost': float('inf'), 'details': None}
        origin_routes, dest_routes_set = node_to_routes_map.get(origin, []), set(node_to_routes_map.get(dest, []))

        for i in set(origin_routes).intersection(dest_routes_set):
            pos1, pos2 = route_node_positions[i].get(origin), route_node_positions[i].get(dest)
            if pos1 is not None and pos2 is not None:
                ivt = abs(route_prefix_times[i][pos2] - route_prefix_times[i][pos1])
                cost = ivt + wait_times[i]
                if cost < best_path['cost']: best_path = {'cost': cost, 'details': ('0_transfer', i, origin, dest)}

        if MAX_TRANSFERS >= 1:
            for r1 in origin_routes:
                for r2 in one_hop_routes[r1].intersection(dest_routes_set):
                    best_transfer_node, min_ivt = None, float('inf')
                    for t_node in transfer_hubs.get((r1, r2), []):
                        pos1_o, pos1_t = route_node_positions[r1].get(origin), route_node_positions[r1].get(t_node)
                        pos2_t, pos2_d = route_node_positions[r2].get(t_node), route_node_positions[r2].get(dest)
                        if all(p is not None for p in [pos1_o, pos1_t, pos2_t, pos2_d]):
                            ivt = abs(route_prefix_times[r1][pos1_t] - route_prefix_times[r1][pos1_o]) + abs(
                                route_prefix_times[r2][pos2_d] - route_prefix_times[r2][pos2_t])
                            if ivt < min_ivt: min_ivt, best_transfer_node = ivt, t_node
                    if best_transfer_node:
                        cost = min_ivt + wait_times[r1] + wait_times[r2] + TRANSFER_PENALTY_TIME
                        if cost < best_path['cost']: best_path = {'cost': cost, 'details': (
                            '1_transfer', r1, r2, origin, dest, best_transfer_node)}

        if best_path['details']:
            path_type, details = best_path['details'][0], best_path['details'][1:]
            served_od_info.append({'O': origin, 'D': dest, 'demand': demand, 'cost': best_path['cost']})
            if path_type == '0_transfer':
                r_idx, o, d = details
                route_segment_loads[r_idx][(o, d)] += demand
            elif path_type == '1_transfer':
                r1, r2, o, d, t_node = details
                route_segment_loads[r1][(o, t_node)] += demand
                route_segment_loads[r2][(t_node, d)] += demand
        else:
            unmet_od_pairs.append({'O': origin, 'D': dest, 'demand': demand})

    return route_segment_loads, unmet_od_pairs, served_od_info


def _get_problematic_od_info(ind, od_df, sp_len, stop_nodes_set, sample_size=10):
    ind_copy = copy.deepcopy(ind)
    if 'frequencies' not in ind_copy or not ind_copy['frequencies']:
        ind_copy['frequencies'] = calculate_demand_responsive_frequencies(ind_copy, od_df, sp_len, stop_nodes_set)

    precomputed = _prepare_network_data_for_individual(ind_copy, sp_len, stop_nodes_set)
    if not precomputed:
        unmet = od_df.rename(columns={'OriginID': 'O', 'DestinationID': 'D', 'PD': 'demand'}).to_dict('records')
        return unmet, []

    _, unmet_pairs, served_info = _calculate_passenger_loads_on_routes(ind_copy, precomputed, od_df, stop_nodes_set)

    unmet_sorted = sorted(unmet_pairs, key=lambda x: x['demand'], reverse=True)
    served_sorted = sorted(served_info, key=lambda x: x['cost'], reverse=True)

    return unmet_sorted[:sample_size], served_sorted[:sample_size]


# --- 5.2 "Fine-tuning" Operators (from experiment.txt, robust logic for small networks) ---
def op_inter_route_swap_stops(ind, **kwargs):
    sp, stop_nodes_set = kwargs['sp'], kwargs['stop_nodes_set']
    ind_copy = copy.deepcopy(ind)
    routes = ind_copy['routes']
    valid_indices = [i for i, r in enumerate(routes) if r and len(r) > 2]
    if len(valid_indices) < 2: return ind
    idx1, idx2 = random.sample(valid_indices, 2)
    r1_original, r2_original = routes[idx1], routes[idx2]
    swappable_indices1 = [i for i, s in enumerate(r1_original) if 0 < i < len(r1_original) - 1 and s in stop_nodes_set]
    swappable_indices2 = [i for i, s in enumerate(r2_original) if 0 < i < len(r2_original) - 1 and s in stop_nodes_set]
    if not swappable_indices1 or not swappable_indices2: return ind
    i1, i2 = random.choice(swappable_indices1), random.choice(swappable_indices2)
    s1, s2 = r1_original[i1], r2_original[i2]
    prev1, next1 = r1_original[i1 - 1], r1_original[i1 + 1]
    path_seg1a, path_seg1b = sp.get(prev1, {}).get(s2), sp.get(s2, {}).get(next1)
    if not path_seg1a or not path_seg1b: return ind
    new_r1 = r1_original[:i1] + path_seg1a[1:-1] + [s2] + path_seg1b[1:] + r1_original[i1 + 2:]
    prev2, next2 = r2_original[i2 - 1], r2_original[i2 + 1]
    path_seg2a, path_seg2b = sp.get(prev2, {}).get(s1), sp.get(s1, {}).get(next2)
    if not path_seg2a or not path_seg2b: return ind
    new_r2 = r2_original[:i2] + path_seg2a[1:-1] + [s1] + path_seg2b[1:] + r2_original[i2 + 2:]
    ind_copy['routes'][idx1], ind_copy['routes'][idx2] = list(dict.fromkeys(new_r1)), list(dict.fromkeys(new_r2))
    return ind_copy


def op_inter_route_relocate_stop(ind, **kwargs):
    sp, stop_nodes_set = kwargs['sp'], kwargs['stop_nodes_set']
    ind_copy = copy.deepcopy(ind)
    routes = ind_copy['routes']
    source_indices = [i for i, r in enumerate(routes) if
                      r and len([s for s in r if s in stop_nodes_set]) > MIN_STOPS_PER_ROUTE]
    dest_indices = [i for i, r in enumerate(routes) if
                    r and len([s for s in r if s in stop_nodes_set]) < MAX_STOPS_PER_ROUTE]
    if not source_indices or not dest_indices: return ind
    idx_source = random.choice(source_indices)
    valid_dest_indices = [i for i in dest_indices if i != idx_source]
    if not valid_dest_indices: return ind
    idx_dest = random.choice(valid_dest_indices)
    r_source, r_dest = routes[idx_source], routes[idx_dest]
    relocatable_indices = [i for i, s in enumerate(r_source) if 0 < i < len(r_source) - 1 and s in stop_nodes_set]
    if not relocatable_indices: return ind
    s_idx = random.choice(relocatable_indices)
    stop_to_move = r_source[s_idx]
    prev_s, next_s = r_source[s_idx - 1], r_source[s_idx + 1]
    path_seg_source = sp.get(prev_s, {}).get(next_s)
    if not path_seg_source: return ind
    new_r_source = r_source[:s_idx] + path_seg_source[1:-1] + r_source[s_idx + 1:]
    if len(r_dest) < 2:
        return ind

    insert_pos = random.randint(1, len(r_dest) - 1)
    prev_d, next_d = r_dest[insert_pos - 1], r_dest[insert_pos]
    path_seg_dest1, path_seg_dest2 = sp.get(prev_d, {}).get(stop_to_move), sp.get(stop_to_move, {}).get(next_d)
    if not path_seg_dest1 or not path_seg_dest2: return ind
    new_r_dest = r_dest[:insert_pos] + path_seg_dest1[1:-1] + [stop_to_move] + path_seg_dest2[1:] + r_dest[
                                                                                                    insert_pos + 1:]
    ind_copy['routes'][idx_source], ind_copy['routes'][idx_dest] = list(dict.fromkeys(new_r_source)), list(
        dict.fromkeys(new_r_dest))
    return ind_copy


def op_intra_route_reroute(ind, **kwargs):
    sp = kwargs['sp']
    ind_copy = copy.deepcopy(ind)
    routes = ind_copy['routes']
    valid_indices = [i for i, r in enumerate(routes) if r and len(r) >= 3]
    if not valid_indices: return ind
    idx = random.choice(valid_indices)
    route = routes[idx]
    start_idx, end_idx = sorted(random.sample(range(len(route)), 2))
    if start_idx + 1 >= end_idx: return ind
    start_node, end_node = route[start_idx], route[end_idx]
    shortest_segment = sp.get(start_node, {}).get(end_node, [])
    if shortest_segment:
        routes[idx] = route[:start_idx] + shortest_segment + route[end_idx + 1:]
    return ind_copy


def op_shrink_route(ind, **kwargs):
    sp, stop_nodes_set = kwargs['sp'], kwargs['stop_nodes_set']
    ind_copy = copy.deepcopy(ind)
    routes = ind_copy['routes']
    eligible_route_indices = [i for i, r in enumerate(routes) if
                              r and len([s for s in r if s in stop_nodes_set]) > MIN_STOPS_PER_ROUTE]
    if not eligible_route_indices: return ind
    route_to_shrink_idx = random.choice(eligible_route_indices)
    route = routes[route_to_shrink_idx]
    if len(route) <= 2: return ind
    stop_idx_to_remove = random.randint(1, len(route) - 2)
    prev_node, next_node = route[stop_idx_to_remove - 1], route[stop_idx_to_remove + 1]
    path_segment = sp.get(prev_node, {}).get(next_node)
    if not path_segment: return ind
    new_route = route[:stop_idx_to_remove] + path_segment[1:]
    ind_copy['routes'][route_to_shrink_idx] = list(dict.fromkeys(new_route))
    return ind_copy


# --- 5.4 【NEW】 Additional Optimization Operators ---

def op_merge_and_split_routes(ind, **kwargs):
    """
    【New Operator - Structural Recombination】
    Merges the two most similar routes in the network, then splits the longest route in the middle.
    This is a structural adjustment operator of medium intensity.
    """
    sp, stop_nodes_set, sp_len = kwargs['sp'], kwargs['stop_nodes_set'], kwargs['sp_len']
    ind_copy = copy.deepcopy(ind)
    routes = ind_copy['routes']
    num_routes = len(routes)
    if num_routes < 3: return ind

    best_pair, max_similarity = (-1, -1), -1.0
    route_sets = [set(r) for r in routes]
    for i in range(num_routes):
        for j in range(i + 1, num_routes):
            if not routes[i] or not routes[j]: continue
            intersection = len(route_sets[i].intersection(route_sets[j]))
            union = len(route_sets[i].union(route_sets[j]))
            if union == 0: continue
            similarity = intersection / union
            if similarity > max_similarity:
                max_similarity, best_pair = similarity, (i, j)

    if max_similarity < 0.1: return ind

    # 2. Merge these two routes
    idx1, idx2 = best_pair
    r1, r2 = routes[idx1], routes[idx2]
    endpoints1, endpoints2 = [r1[0], r1[-1]], [r2[0], r2[-1]]
    min_dist, best_connection = float('inf'), None
    for p1 in endpoints1:
        for p2 in endpoints2:
            dist = sp_len.get(p1, {}).get(p2, float('inf'))
            if dist < min_dist:
                min_dist, best_connection = dist, (p1, p2)

    if not best_connection: return ind

    p1, p2 = best_connection
    connecting_path = sp.get(p1, {}).get(p2, [])
    r1_to_connect = r1 if r1[-1] == p1 else r1[::-1]
    r2_to_connect = r2 if r2[0] == p2 else r2[::-1]
    merged_route = list(dict.fromkeys(r1_to_connect + connecting_path[1:] + r2_to_connect))


    ind_copy['routes'][idx1] = merged_route
    ind_copy['routes'][idx2] = []

    current_routes = ind_copy['routes']
    longest_route_idx, max_len = -1, 0
    for i, r in enumerate(current_routes):
        if r and len(r) > max_len:
            max_len, longest_route_idx = len(r), i

    if longest_route_idx != -1 and max_len > MIN_STOPS_PER_ROUTE * 2:
        route_to_split = current_routes[longest_route_idx]
        split_point = len(route_to_split) // 2

        new_r1 = route_to_split[:split_point]
        new_r2 = route_to_split[split_point:]

        if len([s for s in new_r1 if s in stop_nodes_set]) >= MIN_STOPS_PER_ROUTE and \
                len([s for s in new_r2 if s in stop_nodes_set]) >= MIN_STOPS_PER_ROUTE:

            ind_copy['routes'][longest_route_idx] = new_r1
            empty_idx = next((i for i, r in enumerate(ind_copy['routes']) if not r), -1)
            if empty_idx != -1:
                ind_copy['routes'][empty_idx] = new_r2

    return ind_copy


def op_worst_routes_regeneration(ind, **kwargs):
    """
    【New Operator - Destroy and Rebuild】
    Identifies the route with the least contribution to the network and regenerates a new one using a heuristic method.
    This is a high-intensity exploration operator used to escape local optima.
    """
    sp, stop_nodes_set, od_df, sp_len, G = kwargs['sp'], kwargs['stop_nodes_set'], kwargs['od_df'], kwargs['sp_len'], \
    kwargs['G']
    ind_copy = copy.deepcopy(ind)
    routes = ind_copy['routes']
    if not routes: return ind

    ind_copy_freq = copy.deepcopy(ind)
    ind_copy_freq['frequencies'] = calculate_demand_responsive_frequencies(ind_copy_freq, od_df, sp_len, stop_nodes_set)
    precomputed = _prepare_network_data_for_individual(ind_copy_freq, sp_len, stop_nodes_set)
    if not precomputed: return ind

    segment_loads, _, _ = _calculate_passenger_loads_on_routes(ind_copy_freq, precomputed, od_df, stop_nodes_set)

    route_loads = {i: sum(loads.values()) for i, loads in segment_loads.items()}

    if len(routes) > 1 and route_loads:
        worst_route_idx = min(route_loads, key=route_loads.get)

        new_route = _generate_single_valid_route(sp, stop_nodes_set)
        if new_route:
            ind_copy['routes'][worst_route_idx] = new_route

    return ind_copy

# --- 5.3 "Restructuring" Operators (for medium/large networks) ---
def op_heuristic_swap_stops(ind, **kwargs):
    sp, stop_nodes_set, od_df, sp_len = kwargs['sp'], kwargs['stop_nodes_set'], kwargs['od_df'], kwargs['sp_len']
    ind_copy = copy.deepcopy(ind)
    routes = ind_copy['routes']
    valid_indices = [i for i, r in enumerate(routes) if r and len(r) > 2]
    if len(valid_indices) < 2: return ind

    unmet_ods, long_trip_ods = _get_problematic_od_info(ind, od_df, sp_len, stop_nodes_set)
    problem_points = [od['O'] for od in unmet_ods] + [od['O'] for od in long_trip_ods]
    if not problem_points: return op_inter_route_swap_stops(ind, **kwargs)

    s1, s2 = None, None
    if random.random() < 0.8:
        target_point = random.choice(problem_points)
        min_dist_s1, cand_s1 = float('inf'), None
        for r_idx in valid_indices:
            for stop in routes[r_idx]:
                if stop in stop_nodes_set:
                    dist_to_target = sp_len.get(stop, {}).get(target_point, float('inf'))
                    if dist_to_target < min_dist_s1:
                        min_dist_s1, cand_s1 = dist_to_target, stop
        s1 = cand_s1

        idx2 = random.choice(valid_indices)
        r2_stops = [s for i, s in enumerate(routes[idx2]) if
                    0 < i < len(routes[idx2]) - 1 and s in stop_nodes_set and s != s1]
        if r2_stops: s2 = random.choice(r2_stops)

    if not s1 or not s2: return op_inter_route_swap_stops(ind, **kwargs)

    idx1, i1, idx2, i2 = -1, -1, -1, -1
    for r_idx, r in enumerate(routes):
        if s1 in r: i1, idx1 = r.index(s1), r_idx
        if s2 in r: i2, idx2 = r.index(s2), r_idx

    if not all(x > -1 for x in [idx1, i1, idx2, i2]) or idx1 == idx2 or i1 == 0 or i2 == 0 or i1 >= len(
            routes[idx1]) - 1 or i2 >= len(routes[idx2]) - 1:
        return ind

    r1_original, r2_original = routes[idx1], routes[idx2]
    prev1, next1 = r1_original[i1 - 1], r1_original[i1 + 1]
    path_seg1a, path_seg1b = sp.get(prev1, {}).get(s2), sp.get(s2, {}).get(next1)
    if not path_seg1a or not path_seg1b: return ind
    new_r1 = r1_original[:i1] + path_seg1a[1:-1] + [s2] + path_seg1b[1:-1] + r1_original[i1 + 1:]

    prev2, next2 = r2_original[i2 - 1], r2_original[i2 + 1]
    path_seg2a, path_seg2b = sp.get(prev2, {}).get(s1), sp.get(s1, {}).get(next2)
    if not path_seg2a or not path_seg2b: return ind
    new_r2 = r2_original[:i2] + path_seg2a[1:-1] + [s1] + path_seg2b[1:-1] + r2_original[i2 + 1:]

    ind_copy['routes'][idx1], ind_copy['routes'][idx2] = list(dict.fromkeys(new_r1)), list(dict.fromkeys(new_r2))
    return ind_copy


def op_heuristic_relocate_stop(ind, **kwargs):
    sp, stop_nodes_set, od_df, sp_len = kwargs['sp'], kwargs['stop_nodes_set'], kwargs['od_df'], kwargs['sp_len']
    ind_copy = copy.deepcopy(ind)
    routes = ind_copy['routes']

    source_indices = [i for i, r in enumerate(routes) if
                      r and len([s for s in r if s in stop_nodes_set]) > MIN_STOPS_PER_ROUTE]
    dest_indices = [i for i, r in enumerate(routes) if
                    r and len([s for s in r if s in stop_nodes_set]) < MAX_STOPS_PER_ROUTE]
    if not source_indices or not dest_indices: return ind

    idx_source = random.choice(source_indices)
    valid_dest_indices = [i for i in dest_indices if i != idx_source]
    if not valid_dest_indices: return ind
    idx_dest = random.choice(valid_dest_indices)

    r_source, r_dest = routes[idx_source], routes[idx_dest]
    relocatable_indices = [i for i, s in enumerate(r_source) if 0 < i < len(r_source) - 1 and s in stop_nodes_set]
    if not relocatable_indices: return ind

    s_idx = -1
    unmet_ods, long_trip_ods = _get_problematic_od_info(ind, od_df, sp_len, stop_nodes_set)
    problem_points = [od['O'] for od in unmet_ods] + [od['D'] for od in long_trip_ods]
    if problem_points and random.random() < 0.8:
        target_point = random.choice(problem_points)
        min_dist_to_target, best_stop_idx = float('inf'), -1

        for i in relocatable_indices:
            dist = sp_len.get(r_source[i], {}).get(target_point, float('inf'))
            if dist < min_dist_to_target:
                min_dist_to_target, best_stop_idx = dist, i
        if best_stop_idx != -1: s_idx = best_stop_idx

    if s_idx == -1: s_idx = random.choice(relocatable_indices)

    stop_to_move = r_source[s_idx]
    prev_s, next_s = r_source[s_idx - 1], r_source[s_idx + 1]
    path_seg_source = sp.get(prev_s, {}).get(next_s)
    if not path_seg_source: return ind
    new_r_source = r_source[:s_idx] + path_seg_source[1:-1] + r_source[s_idx + 1:]

    insert_pos = random.randint(1, len(r_dest) - 1)
    prev_d, next_d = r_dest[insert_pos - 1], r_dest[insert_pos]
    path_seg_dest1, path_seg_dest2 = sp.get(prev_d, {}).get(stop_to_move), sp.get(stop_to_move, {}).get(next_d)
    if not path_seg_dest1 or not path_seg_dest2: return ind
    new_r_dest = r_dest[:insert_pos] + path_seg_dest1[1:-1] + [stop_to_move] + path_seg_dest2[1:] + r_dest[insert_pos:]

    ind_copy['routes'][idx_source], ind_copy['routes'][idx_dest] = list(dict.fromkeys(new_r_source)), list(
        dict.fromkeys(new_r_dest))
    return ind_copy


def op_smart_shrink_route(ind, **kwargs):
    sp, stop_nodes_set, od_df, sp_len = kwargs['sp'], kwargs['stop_nodes_set'], kwargs['od_df'], kwargs['sp_len']
    ind_copy = copy.deepcopy(ind)

    ind_copy_freq = copy.deepcopy(ind)
    if 'frequencies' not in ind_copy_freq or not ind_copy_freq['frequencies']:
        ind_copy_freq['frequencies'] = calculate_demand_responsive_frequencies(ind_copy_freq, od_df, sp_len,
                                                                               stop_nodes_set)
    precomputed = _prepare_network_data_for_individual(ind_copy_freq, sp_len, stop_nodes_set)
    if not precomputed: return ind
    segment_loads, _, _ = _calculate_passenger_loads_on_routes(ind_copy_freq, precomputed, od_df, stop_nodes_set)

    stop_usage = collections.defaultdict(int)
    for r_idx in segment_loads:
        for (o, d), demand in segment_loads[r_idx].items():
            stop_usage[o] += demand;
            stop_usage[d] += demand

    eligible_routes = [i for i, r in enumerate(ind_copy['routes']) if
                       r and len([s for s in r if s in stop_nodes_set]) > MIN_STOPS_PER_ROUTE]
    if not eligible_routes: return ind

    route_to_shrink_idx = random.choice(eligible_routes)
    route = ind_copy['routes'][route_to_shrink_idx]

    removable_stops = []
    for i in range(1, len(route) - 1):
        if route[i] in stop_nodes_set:
            removable_stops.append((route[i], stop_usage.get(route[i], 0), i))

    if not removable_stops: return ind

    removable_stops.sort(key=lambda x: x[1])
    _, _, stop_idx_to_remove = removable_stops[0]

    prev_node, next_node = route[stop_idx_to_remove - 1], route[stop_idx_to_remove + 1]
    path_segment = sp.get(prev_node, {}).get(next_node)
    if not path_segment: return ind

    new_route = route[:stop_idx_to_remove] + path_segment[1:-1] + route[stop_idx_to_remove + 1:]
    ind_copy['routes'][route_to_shrink_idx] = list(dict.fromkeys(new_route))
    return ind_copy


def op_smart_extend_route(ind, **kwargs):
    sp, stop_nodes_set, od_df, sp_len = kwargs['sp'], kwargs['stop_nodes_set'], kwargs['od_df'], kwargs['sp_len']
    ind_copy = copy.deepcopy(ind)
    unmet_ods, _ = _get_problematic_od_info(ind, od_df, sp_len, stop_nodes_set, sample_size=5)
    if not unmet_ods: return ind

    best_extension = {'score': -1, 'route_idx': -1, 'new_route': None}

    for target_od in unmet_ods:
        O, D, demand = target_od['O'], target_od['D'], target_od['demand']

        closest_stop, closest_route_idx, closest_endpoint, min_dist = None, -1, None, float('inf')
        for i, r in enumerate(ind_copy['routes']):
            if not r or len(r) >= MAX_STOPS_PER_ROUTE * 0.9: continue

            for endpoint in [r[0], r[-1]]:
                dist = sp_len.get(endpoint, {}).get(O, float('inf'))
                if dist < min_dist:
                    min_dist, closest_route_idx, closest_endpoint = dist, i, endpoint

        if closest_route_idx != -1:
            path_to_o = sp.get(closest_endpoint, {}).get(O)
            if not path_to_o: continue

            cost = get_route_travel_time(path_to_o, sp_len)
            score = demand / (1 + cost)

            if score > best_extension['score']:
                route_to_extend = ind_copy['routes'][closest_route_idx]
                if closest_endpoint == route_to_extend[0]:
                    new_route = path_to_o[::-1][:-1] + route_to_extend
                else:
                    new_route = route_to_extend + path_to_o[1:]
                best_extension = {'score': score, 'route_idx': closest_route_idx,
                                  'new_route': list(dict.fromkeys(new_route))}

    if best_extension['route_idx'] != -1:
        ind_copy['routes'][best_extension['route_idx']] = best_extension['new_route']

    return ind_copy


def op_merge_and_regenerate_routes(ind, **kwargs):
    sp, stop_nodes_set, sp_len = kwargs['sp'], kwargs['stop_nodes_set'], kwargs['sp_len']
    ind_copy = copy.deepcopy(ind)
    routes = ind_copy['routes']
    num_routes = len(routes)
    if num_routes < 2: return ind

    best_pair, max_similarity = (-1, -1), -1.0
    for i in range(num_routes):
        for j in range(i + 1, num_routes):
            if not routes[i] or not routes[j]: continue
            set1, set2 = set(routes[i]), set(routes[j])
            intersection, union = len(set1.intersection(set2)), len(set1.union(set2))
            if union == 0: continue
            similarity = intersection / union
            if similarity > max_similarity:
                max_similarity, best_pair = similarity, (i, j)

    if max_similarity < 0.3: return ind

    idx1, idx2 = best_pair
    r1, r2 = routes[idx1], routes[idx2]

    endpoints1, endpoints2 = [r1[0], r1[-1]], [r2[0], r2[-1]]
    min_dist, best_connection = float('inf'), None
    for p1 in endpoints1:
        for p2 in endpoints2:
            dist = sp_len.get(p1, {}).get(p2, float('inf'))
            if dist < min_dist:
                min_dist, best_connection = dist, (p1, p2)

    if not best_connection: return ind

    p1, p2 = best_connection
    connecting_path = sp.get(p1, {}).get(p2, [])
    r1_to_connect = r1 if r1[-1] == p1 else r1[::-1]
    r2_to_connect = r2 if r2[0] == p2 else r2[::-1]
    merged_route = r1_to_connect + connecting_path[1:-1] + r2_to_connect

    ind_copy['routes'][idx1] = list(dict.fromkeys(merged_route))
    ind_copy['routes'][idx2] = _generate_single_valid_route(sp, stop_nodes_set)

    return ind_copy


def op_smart_split_route(ind, **kwargs):
    sp, stop_nodes_set, od_df, sp_len = kwargs['sp'], kwargs['stop_nodes_set'], kwargs['od_df'], kwargs['sp_len']
    ind_copy = copy.deepcopy(ind)

    if not any(ind_copy['routes']): return ind
    candidate_routes = []
    avg_len = np.mean([len(r) for r in ind_copy['routes'] if r])
    for i, r in enumerate(ind_copy['routes']):
        if r and len(r) > avg_len * 1.2:
            candidate_routes.append(i)
    if not candidate_routes: return ind

    route_to_split_idx = random.choice(candidate_routes)
    route = ind_copy['routes'][route_to_split_idx]

    ind_copy_freq = copy.deepcopy(ind)
    ind_copy_freq['frequencies'] = calculate_demand_responsive_frequencies(ind_copy_freq, od_df, sp_len, stop_nodes_set)
    precomputed = _prepare_network_data_for_individual(ind_copy_freq, sp_len, stop_nodes_set)
    if not precomputed: return ind
    segment_loads, _, _ = _calculate_passenger_loads_on_routes(ind_copy_freq, precomputed, od_df, stop_nodes_set)

    pos_map = {node: pos for pos, node in enumerate(route)}
    load_deltas = collections.defaultdict(int)
    if route_to_split_idx in segment_loads:
        for (o, d), demand in segment_loads[route_to_split_idx].items():
            pos_o, pos_d = pos_map.get(o), pos_map.get(d)
            if pos_o is not None and pos_d is not None:
                start, end = min(pos_o, pos_d), max(pos_o, pos_d)
                if start < end: load_deltas[start] += demand; load_deltas[end] -= demand

    loads = [0] * (len(route) - 1)
    current_load = 0
    for i in range(len(route) - 1):
        current_load += load_deltas.get(i, 0)
        loads[i] = current_load

    if len(loads) < MIN_STOPS_PER_ROUTE * 2: return ind
    split_range = range(MIN_STOPS_PER_ROUTE, len(loads) - MIN_STOPS_PER_ROUTE)
    if not split_range: return ind

    min_load_idx, min_load = -1, float('inf')
    for i in split_range:
        if loads[i] < min_load:
            min_load, min_load_idx = loads[i], i
    if min_load_idx == -1: return ind
    split_point_idx = min_load_idx + 1

    route1, route2 = route[:split_point_idx], route[split_point_idx:]

    stops1 = len([s for s in route1 if s in stop_nodes_set])
    stops2 = len([s for s in route2 if s in stop_nodes_set])
    if stops1 < MIN_STOPS_PER_ROUTE or stops2 < MIN_STOPS_PER_ROUTE: return ind

    ind_copy['routes'][route_to_split_idx] = route1

    worst_route_idx, min_load_sum = -1, float('inf')
    for i, r in enumerate(ind_copy['routes']):
        if i == route_to_split_idx: continue
        load_sum = sum(segment_loads.get(i, {}).values())
        if load_sum < min_load_sum:
            min_load_sum, worst_route_idx = load_sum, i

    if worst_route_idx != -1:
        ind_copy['routes'][worst_route_idx] = route2

    return ind_copy


# --- 6. Main Process ---
def run_genetic_algorithm(G, od_df, sp_len, sp, initial_solution_path, stop_nodes_set, algorithm_mode='MA-RL',
                          network_name='mandl'):
    output_path = os.path.join(OUTPUT_PATH_BASE, network_name, algorithm_mode)
    if not os.path.exists(output_path): os.makedirs(output_path)
    print(f"\n======== Starting {algorithm_mode} Algorithm Optimization (Results will be saved to: {output_path}) ========")

    population = []
    initial_solution = load_initial_solution(initial_solution_path)
    if initial_solution: population.append(initial_solution)
    initial_to_generate = POP_SIZE - len(population)
    if initial_to_generate > 0:
        print(f"Generating {initial_to_generate} initial individuals using heuristic method...")
        for _ in tqdm(range(initial_to_generate), desc="Generating Heuristic Individuals"):
            population.append(create_heuristic_individual(od_df, sp, sp_len, G, stop_nodes_set=stop_nodes_set))

    # --- Dynamic Operator Selection ---
    # --- [MODIFIED]: Assigning operators from experiment.txt for small networks (Mandl) ---
    operators_fine_tuning = [
        op_intra_route_reroute,         # a0: Fine-tuning
        op_inter_route_relocate_stop,   # a1: Local adjustment
        op_merge_and_split_routes,      # a2: Structural recombination
        op_worst_routes_regeneration    # a3: Destroy and rebuild
    ]
    names_fine_tuning = [
        "Intra-Route Reroute",
        "Inter-Route Relocate",
        "Merge & Split",
        "Destroy & Rebuild"
    ]

    # --- Toolbox 2: For medium to large networks (like Mumford), focuses on restructuring, uses advanced operators ---
    operators_restructuring = [
        op_heuristic_swap_stops,
        op_heuristic_relocate_stop,
        op_smart_shrink_route,
        op_smart_extend_route,
        op_merge_and_regenerate_routes,
        op_smart_split_route
    ]
    names_restructuring = [
        "Heuristic Swap",
        "Heuristic Relocate",
        "Smart Shrink",
        "Smart Extend",
        "Merge & Regenerate",
        "Smart Split"
    ]

    if network_name == 'mandl':
        print("\nSmall network (Mandl) detected, loading [Fine-Tuning] operator toolbox...")
        action_operators = operators_fine_tuning
        action_names = names_fine_tuning
    else:  # Mumford0 or other
        print("\nMedium/Large network detected, loading [Restructuring] operator toolbox...")
        action_operators = operators_restructuring
        action_names = names_restructuring

    operator_kwargs = {'G': G, 'od_df': od_df, 'sp_len': sp_len, 'sp': sp, 'stop_nodes_set': stop_nodes_set}
    rl_tuner = None
    if algorithm_mode == 'MA-RL':
        rl_tuner = Advanced_RL_Tuner(actions=action_names, stagnation_threshold=RL_STAGNATION_THRESHOLD,
                                     temperature=0.1)
        print("Enabled [Advanced MA-RL Mode].")

    operator_stats = None
    if algorithm_mode == 'MA-Random':
        print("Enabled [Operator Performance Analysis Mode].")
        operator_stats = {name: {"success_count": 0, "total_improvement": 0.0, "use_count": 0} for name in action_names}

    operator_selection_history = {}
    q_table_snapshots = {}
    reward_dynamics_history = []
    learning_records = []
    best_fitness_history, best_individual_routes, best_fitness = [], None, float('inf')
    start_time = time.time()
    total_demand = od_df['PD'].sum()
    manager = Manager()
    fitness_cache = manager.dict()
    fitness_calculator_partial = partial(calculate_fitness_wrapper, sp_len=sp_len, stop_nodes_set=stop_nodes_set,
                                         od_df=od_df, G=G, sp=sp, fitness_cache=fitness_cache,
                                         total_demand=total_demand)

    print(f"Parallel computation started, using {os.cpu_count()} CPU cores.")
    with Pool(os.cpu_count()) as pool:
        with tqdm(total=MAX_GENERATIONS, desc=f"Running {algorithm_mode}") as pbar:
            for gen in range(MAX_GENERATIONS):
                fitnesses = pool.map(fitness_calculator_partial, population)
                current_best_idx = np.argmin(fitnesses)
                if fitnesses[current_best_idx] < best_fitness:
                    best_fitness = fitnesses[current_best_idx]
                    best_individual_routes = copy.deepcopy(population[current_best_idx])
                if best_individual_routes:
                    temp_best_complete = copy.deepcopy(best_individual_routes)
                    demand_freq = calculate_demand_responsive_frequencies(temp_best_complete, od_df, sp_len,
                                                                          stop_nodes_set)
                    temp_best_complete['frequencies'] = demand_freq
                    valid_best = repair_and_validate_individual(temp_best_complete, G, sp_len, sp, stop_nodes_set)
                    best_precomputed_data = _prepare_network_data_for_individual(valid_best, sp_len, stop_nodes_set)
                    best_metrics = calculate_performance_metrics(best_precomputed_data, od_df)
                    unmet_passengers = best_metrics.get('unmet_passengers', 0)
                    direct_rate = (best_metrics.get('passengers_0_transfer',
                                                    0) / total_demand * 100) if total_demand > 0 else 0
                    pbar.set_postfix(best_fitness=f'{best_fitness:.2f}', unmet=f"{unmet_passengers:.0f}",
                                     direct_rate=f"{direct_rate:.1f}%", cache_size=len(fitness_cache))
                best_fitness_history.append(best_fitness)
                sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda p: p[0])]
                next_population = sorted_population[:int(POP_SIZE * ELITISM_RATE)]
                selected_parents = selection(population, fitnesses)
                children_to_process = []
                while len(next_population) + len(children_to_process) < POP_SIZE:
                    p1, p2 = random.sample(selected_parents, 2)
                    c1, c2 = crossover(p1, p2)
                    children_to_process.append(mutate(c1, G, sp, stop_nodes_set))
                    if len(next_population) + len(children_to_process) < POP_SIZE:
                        children_to_process.append(mutate(c2, G, sp, stop_nodes_set))

                if algorithm_mode == 'GA':
                    next_population.extend(children_to_process)
                else:
                    if algorithm_mode == 'MA-RL' and rl_tuner:
                        current_state = rl_tuner.get_state(best_fitness, fitnesses)
                        if gen > 0 and learning_records:
                            next_state_for_update = current_state
                            for record in learning_records:
                                rl_tuner.update_q_table(record['state'], record['action'], record['reward'],
                                                        next_state_for_update)
                            learning_records = []

                        fitnesses_before = pool.map(fitness_calculator_partial, children_to_process)
                        refined_children, actions_chosen_this_gen = [], []
                        for i, child in enumerate(children_to_process):
                            if random.random() < rl_tuner.epsilon:
                                action_idx = random.randint(0, rl_tuner.num_actions - 1)
                            else:
                                action_probabilities = rl_tuner.choose_action_probabilities(current_state)
                                action_idx = np.random.choice(rl_tuner.num_actions, p=action_probabilities)
                            actions_chosen_this_gen.append(action_idx)
                            operator_to_apply = action_operators[action_idx]
                            refined_children.append(operator_to_apply(copy.deepcopy(child), **operator_kwargs))

                        operator_selection_history[gen] = actions_chosen_this_gen
                        fitnesses_after = pool.map(fitness_calculator_partial, refined_children)

                        for i in range(len(children_to_process)):
                            improvement = fitnesses_before[i] - fitnesses_after[i]
                            reward = np.tanh(improvement / (fitnesses_before[i] + 1e-6))
                            action_idx = actions_chosen_this_gen[i]
                            learning_records.append({'state': current_state, 'action': action_idx, 'reward': reward})
                            reward_dynamics_history.append({'gen': gen, 'action': action_idx, 'reward': reward})

                        next_population.extend(refined_children)

                    elif algorithm_mode == 'MA-Random':
                        actions_chosen_random = [random.randint(0, len(action_operators) - 1) for _ in
                                                 children_to_process]
                        fitnesses_before = pool.map(fitness_calculator_partial, children_to_process)
                        refined_children = []
                        for i, child in enumerate(children_to_process):
                            action_idx = actions_chosen_random[i]
                            operator_name = action_names[action_idx]
                            refined_children.append(
                                action_operators[action_idx](copy.deepcopy(child), **operator_kwargs))
                            operator_stats[operator_name]["use_count"] += 1
                        fitnesses_after = pool.map(fitness_calculator_partial, refined_children)
                        for i in range(len(children_to_process)):
                            improvement = fitnesses_before[i] - fitnesses_after[i]
                            if improvement > 1e-6:
                                operator_name = action_names[actions_chosen_random[i]]
                                operator_stats[operator_name]["success_count"] += 1
                                operator_stats[operator_name]["total_improvement"] += improvement
                        next_population.extend(refined_children)

                    elif algorithm_mode == 'MA-Fixed':
                        refined_children = []
                        for child in children_to_process:
                            operator_to_apply = random.choice(action_operators)
                            refined_children.append(operator_to_apply(copy.deepcopy(child), **operator_kwargs))
                        next_population.extend(refined_children)

                population = next_population[:POP_SIZE]
                if algorithm_mode == 'MA-RL' and rl_tuner:
                    if gen in [10, 60, 100, MAX_GENERATIONS-1]:
                        q_table_snapshots[gen] = copy.deepcopy(rl_tuner.q_table)
                if rl_tuner: rl_tuner.decay_epsilon()
                pbar.update(1)

    total_time = time.time() - start_time
    if operator_stats:
        print("\n" + "=" * 25 + " Operator Performance Analysis Report (MA-Random Mode) " + "=" * 25)
        stats_list = []
        for name, data in operator_stats.items():
            success_rate = (data["success_count"] / data["use_count"] * 100) if data["use_count"] > 0 else 0
            avg_improvement = (data["total_improvement"] / data["success_count"]) if data["success_count"] > 0 else 0
            stats_list.append({"Operator": name, "Use Count": data["use_count"], "Success Count": data["success_count"],
                               "Success Rate (%)": f"{success_rate:.2f}",
                               "Total Improvement": f"{data['total_improvement']:.2f}",
                               "Average Improvement": f"{avg_improvement:.2f}"})
        stats_df = pd.DataFrame(stats_list)
        print(stats_df.to_string(index=False))
        analysis_path = os.path.join(output_path, "operator_performance_analysis.csv")
        stats_df.to_csv(analysis_path, index=False, encoding='utf-8-sig')
        print(f"\nAnalysis report saved to: {analysis_path}")
        print("=" * 75)

    final_best_individual = copy.deepcopy(best_individual_routes) if best_individual_routes else {'routes': []}
    final_frequencies = calculate_demand_responsive_frequencies(final_best_individual, od_df, sp_len, stop_nodes_set)
    final_best_individual['frequencies'] = final_frequencies
    final_best_individual = repair_and_validate_individual(final_best_individual, G, sp_len, sp, stop_nodes_set)
    best_precomputed_data = _prepare_network_data_for_individual(final_best_individual, sp_len, stop_nodes_set)
    best_metrics = calculate_performance_metrics(best_precomputed_data, od_df)
    print(f"\n======== {algorithm_mode} algorithm finished. Total time: {total_time / 60:.2f} minutes.========")
    print(f"Final Best Fitness: {best_metrics.get('fitness', 'N/A'):.2f}")
    print(f"Unmet Passengers: {best_metrics.get('unmet_passengers', 'N/A')}")
    print(f"Final Cache Size: {len(fitness_cache)} (total unique individuals evaluated)")
    if rl_tuner:
        q_table_df_path = os.path.join(output_path, "q_table_final.csv")
        q_table_labels = ["Stagnation-HighDiversity", "Stagnation-LowDiversity", "Convergence-HighDiversity", "Convergence-LowDiversity"]
        pd.DataFrame(rl_tuner.q_table, columns=rl_tuner.actions, index=q_table_labels).to_csv(q_table_df_path,
                                                                                              encoding='utf-8-sig')
        print(f"Final Q-Table saved to: {q_table_df_path}")
    return final_best_individual, best_fitness_history, rl_tuner, output_path, operator_selection_history, q_table_snapshots, reward_dynamics_history


# --- 7. Results Analysis and Saving ---
def plot_operator_selection_dynamics(history, action_aliases, filename):
    if not history:
        print("Warning: Operator selection history is empty, cannot generate dynamics plot.")
        return

    plt.figure(figsize=(15, 8))
    gens, actions = [], []
    for gen, action_indices in history.items():
        for action_idx in action_indices:
            gens.append(gen)
            actions.append(action_idx)

    y_jitter = np.random.normal(0, 0.08, size=len(actions))
    plt.scatter(gens, np.array(actions) + y_jitter, alpha=0.6, s=15, c=actions, cmap='viridis')

    plt.title("RL Controller Operator Selection Dynamics", fontsize=24)
    plt.xlabel("Generation", fontsize=18)
    plt.ylabel("Selected Local Search Operator", fontsize=18)
    plt.yticks(ticks=range(len(action_aliases)), labels=action_aliases)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_q_table_heatmap_revised(q_table, actions, states, filename):
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(
        q_table,
        annot=True, fmt=".3f", cmap="viridis", 
        xticklabels=actions, yticklabels=states,
        linewidths=.5,
        linecolor='white',
        cbar_kws={"shrink": .82}, 
        annot_kws={"size": 14, "weight": "bold"} 
    )
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=0, fontsize=14)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=14)
    plt.title("Q-Table Heatmap", fontsize=22, pad=20)
    plt.xlabel("Local Search Operators", fontsize=16)
    plt.ylabel("Population State", fontsize=16)
    plt.tight_layout(pad=1.5)
    plt.savefig(filename, dpi=500)
    plt.close()

def display_and_save_results(best_individual, history, G, od_df, sp_len, stop_nodes_set, output_path, rl_tuner=None,
                             operator_history=None, q_snapshots=None, dynamics_history=None):
    print("\n--- Generating and saving final optimization results ---")
    if not os.path.exists(output_path): os.makedirs(output_path)
    plt.rcParams['axes.unicode_minus'] = False

    best_precomputed = _prepare_network_data_for_individual(best_individual, sp_len, stop_nodes_set)
    details = calculate_performance_metrics(best_precomputed, od_df)
    total_demand = od_df['PD'].sum()
    served_demand = total_demand - details.get('unmet_passengers', 0)
    total_fleet = 0
    if best_individual and best_individual.get('routes') and best_individual.get('frequencies'):
        routes, frequencies = best_individual['routes'], best_individual['frequencies']
        if routes and frequencies:
            round_trip_times = [2 * get_route_travel_time(r, sp_len) for r in routes]
            valid_routes_data = [(rtt, f) for rtt, f in zip(round_trip_times, frequencies) if not math.isinf(rtt)]
            total_fleet = sum([math.ceil((rtt * f) / 60) for rtt, f in valid_routes_data])

    print("\n" + "=" * 25 + " Final Solution Performance Report " + "=" * 25)
    final_num_routes = len(best_individual.get('routes', []))
    print(f"{'Metric':<40} | {'Value'}")
    print("-" * 60)
    print(f"{'Final Fitness':<40} | {details.get('fitness', 'N/A'):.2f}")
    print(f"{'Total Passenger Time Cost':<40} | {details.get('total_passenger_time_cost', 0):.2f}")
    print(f"{'  - Total In-Vehicle Time':<39} | {details.get('total_in_vehicle_time', 0):.2f}")
    print(f"{'  - Total Wait Time':<39} | {details.get('total_wait_time', 0):.2f}")
    print(f"{'  - Total Transfer Penalty Time':<39} | {details.get('total_transfer_time', 0):.2f}")
    print(f"{'Unmet Passengers':<40} | {details.get('unmet_passengers', 'N/A')} / {total_demand} ({details.get('unmet_passengers', 0) / total_demand * 100:.2f}%)")
    print(f"{'Demand Satisfaction Rate (%)':<40} | {(served_demand / total_demand * 100):.2f}%" if total_demand > 0 else "0.00%")
    print(f"{'Avg. Travel Time per Served Passenger':<40} | {(details.get('total_passenger_time_cost', 0) / served_demand):.2f}" if served_demand > 0 else "N/A")
    print(f"{'Passengers (0 Transfers)':<40} | {details.get('passengers_0_transfer', 0)} ({details.get('passengers_0_transfer', 0) / total_demand * 100:.2f}%)")
    print(f"{'Passengers (1 Transfer)':<40} | {details.get('passengers_1_transfer', 0)} ({details.get('passengers_1_transfer', 0) / total_demand * 100:.2f}%)")
    print(f"{'Passengers (2 Transfers)':<40} | {details.get('passengers_2_transfer', 0)} ({details.get('passengers_2_transfer', 0) / total_demand * 100:.2f}%)")
    print(f"{'Final Number of Routes':<40} | {final_num_routes} / {NUM_ROUTES}")
    print(f"{'Total Fleet Size':<40} | {total_fleet} / {MAX_TOTAL_FLEET}")
    print("=" * 60 + "\n")

    summary_df = pd.DataFrame({
        "Metric": ["Final Fitness", "Total Passenger Time Cost", " - Total In-Vehicle Time", " - Total Wait Time", " - Total Transfer Penalty Time",
                 "Unmet Passengers", "Total Demand", "Demand Satisfaction Rate (%)", "Passengers (0 Transfers)", "Passengers (1 Transfer)", "Passengers (2 Transfers)",
                 "Direct Transfer Rate (%)", "1-Transfer Rate (%)", "2-Transfer Rate (%)", "Avg. Travel Time per Served Passenger", "Final Number of Routes",
                 "Total Fleet Size", "Max Fleet Size (Constraint)"],
        "Value": [f"{details.get('fitness', 0):.2f}", f"{details.get('total_passenger_time_cost', 0):.2f}",
                 f"{details.get('total_in_vehicle_time', 0):.2f}", f"{details.get('total_wait_time', 0):.2f}",
                 f"{details.get('total_transfer_time', 0):.2f}", f"{details.get('unmet_passengers', 0)}",
                 f"{total_demand}", f"{(served_demand / total_demand * 100):.2f}" if total_demand > 0 else "0.00",
                 f"{details.get('passengers_0_transfer', 0)}", f"{details.get('passengers_1_transfer', 0)}",
                 f"{details.get('passengers_2_transfer', 0)}",
                 f"{(details.get('passengers_0_transfer', 0) / total_demand * 100):.2f}" if total_demand > 0 else "0.00",
                 f"{(details.get('passengers_1_transfer', 0) / total_demand * 100):.2f}" if total_demand > 0 else "0.00",
                 f"{(details.get('passengers_2_transfer', 0) / total_demand * 100):.2f}" if total_demand > 0 else "0.00",
                 f"{(details.get('total_passenger_time_cost', 0) / served_demand):.2f}" if served_demand > 0 else "N/A",
                 final_num_routes, f"{total_fleet}", MAX_TOTAL_FLEET]
    })
    summary_df.to_csv(os.path.join(output_path, "summary_metrics.csv"), index=False, encoding='utf-8-sig')
    print(f"Key metrics saved to: {os.path.join(output_path, 'summary_metrics.csv')}")

    if best_individual and best_individual.get('routes'):
        route_details = []
        for i, r in enumerate(best_individual['routes']):
            rtt = 2 * get_route_travel_time(r, sp_len)
            fleet = math.ceil((rtt * best_individual['frequencies'][i]) / 60) if not math.isinf(rtt) else 'N/A'
            route_details.append({"Route ID": i + 1, "Number of Stops": len([n for n in r if n in stop_nodes_set]),
                                  "Frequency (buses/hr)": best_individual['frequencies'][i],
                                  "Round Trip Time (min)": f"{rtt:.2f}" if not math.isinf(rtt) else "Infinite",
                                  "Required Fleet Size": fleet})
        pd.DataFrame(route_details).to_csv(os.path.join(output_path, "best_routes_details.csv"), index=False,
                                           encoding='utf-8-sig')
        print(f"Route details saved to: {os.path.join(output_path, 'best_routes_details.csv')}")
        sequences = [
            {"routeID": i + 1, "node_seq": ",".join(map(str, r)), "frequency": best_individual['frequencies'][i]} for
            i, r in enumerate(best_individual['routes'])]
        pd.DataFrame(sequences).to_csv(os.path.join(output_path, "best_routes_sequence.csv"), index=False,
                                       encoding='utf-8-sig')
        print(f"Route sequences saved to: {os.path.join(output_path, 'best_routes_sequence.csv')}")

    plt.figure(figsize=(12, 7))
    plt.plot(history, linestyle='-', linewidth=3)
    plt.title("Fitness Evolution Curve", fontsize=24)
    plt.xlabel("Generation", fontsize=18)
    plt.ylabel("Best Fitness", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "fitness_evolution.png"))
    plt.close()
    print(f"Fitness curve plot saved to: {os.path.join(output_path, 'fitness_evolution.png')}")

    if rl_tuner:
        print("\n--- Generating visualizations for the final Q-Table... ---")
        q_table = rl_tuner.q_table
        original_actions = rl_tuner.actions
        action_aliases = [f'a{i}' for i in range(len(original_actions))]

        print("Plot Operator Alias Legend:")
        for alias, original in zip(action_aliases, original_actions):
            print(f"  {alias}: {original}")
        states_for_plot = ["Z0", "Z1", "Z2", "Z3"]
        plot_q_table_heatmap_revised(q_table, action_aliases, states_for_plot, os.path.join(output_path, "q_table_heatmap.png"))
        print(f"Q-Table heatmap saved to: {os.path.join(output_path, 'q_table_heatmap.png')}")

        if operator_history:
            print("\n--- Generating operator selection dynamics plot... ---")
            plot_operator_selection_dynamics(
                operator_history,
                action_aliases,
                os.path.join(output_path, "operator_selection_dynamics.png")
            )
            print(f"Operator selection dynamics plot saved to: {os.path.join(output_path, 'operator_selection_dynamics.png')}")

    print("\n--- Saving comprehensive analysis data for external plotting ---")
    analysis_data = {
        'q_table_snapshots': q_snapshots,
        'reward_dynamics_history': dynamics_history,
        'action_names': rl_tuner.actions if rl_tuner else []
    }

    data_filepath = os.path.join(output_path, "comprehensive_analysis_data.pkl")
    with open(data_filepath, 'wb') as f:
        pickle.dump(analysis_data, f)
    print(f"Comprehensive data for plotting saved to: {data_filepath}")

# --- 8. Main Program Entry ---
def main():
    BASE_DIR = Path(__file__).resolve().parent
    NETWORK_CONFIGS = {
        'mandl': {
            "path": BASE_DIR / "data" / "mandl",
            "initial_solution": "route_node.csv",
            "num_routes": 4, "max_fleet": 99, "max_stops": 15, "min_stops": 3
        },
        'mumford0': {
            "path": BASE_DIR / "data" / "mumford0",
            "initial_solution": "route_node.csv",
            "num_routes": 12, "max_fleet": 150, "max_stops": 20, "min_stops": 5
        }
    }
    print("======================================================")
    print("=      Bus Network Optimization Algorithm Platform (V5.1 Integrated)      =")
    print("======================================================")
    print("Step 1: Please select the benchmark network to test:")
    print("  1. Mandl's Swiss Network (15 nodes)")
    print("  2. Mumford0 Network (30 nodes)")
    print("======================================================")

    network_map = {'1': 'mandl', '2': 'mumford0'}
    net_choice = ''
    while net_choice not in network_map:
        net_choice = input("Enter network option (1-2): ")
    selected_network_name = network_map[net_choice]

    config = NETWORK_CONFIGS[selected_network_name]
    global BASE_PATH, NUM_ROUTES, MAX_TOTAL_FLEET, MAX_STOPS_PER_ROUTE, MIN_STOPS_PER_ROUTE
    BASE_PATH, NUM_ROUTES, MAX_TOTAL_FLEET, MAX_STOPS_PER_ROUTE, MIN_STOPS_PER_ROUTE = \
        config["path"], config["num_routes"], config["max_fleet"], config["max_stops"], config["min_stops"]
    initial_solution_filename = config["initial_solution"]
    print(f"\nNetwork selected: {selected_network_name.upper()}. Parameters updated:")
    print(f"  - Data Path: {BASE_PATH}\n  - Number of Routes: {NUM_ROUTES}\n  - Max Fleet Size: {MAX_TOTAL_FLEET}")
    print("-" * 50)

    print("\nStep 2: Please select the algorithm mode to run:")
    print("  1. GA (Standard Genetic Algorithm)")
    print("  2. MA-Random (Memetic Algorithm with Random Local Search and Performance Analysis)")
    print("  3. MA-Fixed (Memetic Algorithm with Fixed-Policy Local Search)")
    print("  4. MA-RL (【Recommended】Advanced Memetic Algorithm with Reinforcement Learning)")
    print("======================================================")

    mode_map = {'1': 'GA', '2': 'MA-Random', '3': 'MA-Fixed', '4': 'MA-RL'}
    algo_choice = ''
    while algo_choice not in mode_map:
        algo_choice = input("Enter algorithm option (1-4): ")
    selected_mode = mode_map[algo_choice]

    if not os.path.exists(BASE_PATH):
        print(f"\nError: Base data path '{BASE_PATH}' does not exist.")
        return

    G, node_df, od_df, sp_len, sp, stop_nodes_set = load_data(BASE_PATH)
    if G is not None and stop_nodes_set is not None:
        initial_solution_file = os.path.join(BASE_PATH,
                                             initial_solution_filename) if initial_solution_filename else None
        if initial_solution_file and not os.path.exists(initial_solution_file):
            print(f"Warning: Initial solution file '{initial_solution_file}' not found. A fully random initial population will be generated.")
            initial_solution_file = None

        best_solution, fitness_history, final_rl_tuner, output_path, operator_history, q_snapshots, dynamics_history = run_genetic_algorithm(
            G, od_df, sp_len, sp, initial_solution_file, stop_nodes_set,
            algorithm_mode=selected_mode, network_name=selected_network_name
        )

        if best_solution:
            display_and_save_results(best_solution, fitness_history, G, od_df, sp_len, stop_nodes_set, output_path,
                                     final_rl_tuner, operator_history, q_snapshots, dynamics_history)
        else:
            print("Could not find a valid solution.")


if __name__ == '__main__':
    freeze_support()
    main()