#!/usr/bin/env python3
import os
import sys
import time
import numpy as np

import traci

# Use the exact logic from our PoC for the comparison
def compute_weight(dist, aoi, is_icv, risk_aware=True, risk_score=0.0):
    if dist > 300: return -np.inf
    cq = max(0, 1.0 - (dist/300)**2)
    su = 1.0 - np.exp(-aoi)
    if risk_aware:
        rm = 1.0 + 3.0 * risk_score
        return cq * su * rm
    else:
        return cq * su * (1.5 if is_icv else 1.0)

def greedy_matching(icvs, nicvs, risk_aware=True):
    # icvs and nicvs are dicts: id -> dict(pos, aoi, risk_score, is_icv)
    if not icvs or not nicvs: return {}
    
    nicv_list = list(nicvs.items())
    if risk_aware:
        nicv_list.sort(key=lambda x: x[1]['risk_score'], reverse=True)
        
    avail_icvs = set(icvs.keys())
    assignment = {}
    
    for j_id, j_data in nicv_list:
        best_w = -np.inf
        best_i = None
        for i_id in avail_icvs:
            i_data = icvs[i_id]
            dist = np.sqrt((i_data['pos'][0]-j_data['pos'][0])**2 + (i_data['pos'][1]-j_data['pos'][1])**2)
            w = compute_weight(dist, j_data['aoi'], j_data['is_icv'], risk_aware, j_data['risk_score'])
            if w > best_w:
                best_w = w
                best_i = i_id
        if best_i is not None and best_w > 0:
            assignment[best_i] = j_id
            avail_icvs.remove(best_i)
            
    return assignment

def waterfill_allocation(vehicles, total_bw, risk_aware=True):
    n = len(vehicles)
    if n == 0: return {}
    
    veh_list = list(vehicles.items())
    # Channel gains (simulated Rayleigh)
    h = np.random.exponential(1.0, n)
    
    weights = np.zeros(n)
    for idx, (v_id, v_data) in enumerate(veh_list):
        if risk_aware:
            weights[idx] = (1.0 + 3.0 * v_data['risk_score']) / (v_data['aoi'] + 0.1)
        else:
            weights[idx] = (1.5 if v_data['is_icv'] else 1.0) / (v_data['aoi'] + 0.1)
            
    eff = h * weights
    allocated = np.ones(n) * 0.1
    remaining = total_bw - np.sum(allocated)
    
    if remaining > 0:
        sorted_idx = np.argsort(-eff)
        for _ in range(3):
            for idx in sorted_idx:
                alloc = remaining * eff[idx] / (np.sum(eff) + 1e-10) + 0.1
                allocated[idx] = max(0.1, alloc)
            allocated *= total_bw / max(np.sum(allocated), 1e-10)
            
    return {veh_list[i][0]: allocated[i] for i in range(n)}

def sca_allocation(vehicles, total_bw):
    # Simulated iterative SCA from our PoC. For real-time simulation we use a simplified proxy 
    # to emulate the base paper's risk-blind bandwidth distribution.
    return waterfill_allocation(vehicles, total_bw, risk_aware=False)

def main():
    print("Starting SUMO TraCI Simulation...")
    traci.start(["sumo", "-c", "highway.sumocfg", "--no-warnings"])
    
    total_bw = 100.0
    step = 0
    
    # State tracking
    vehicles_state = {}
    
    # Metrics
    base_aoi_history = []
    ours_aoi_history = []
    base_bw_danger = []
    ours_bw_danger = []
    
    # Sim loop (simulate 100 seconds)
    while traci.simulation.getTime() < 100:
        traci.simulationStep()
        
        veh_ids = traci.vehicle.getIDList()
        if not veh_ids:
            continue
            
        current_vehs = {}
        icvs = {}
        nicvs = {}
        
        for vid in veh_ids:
            pos = traci.vehicle.getPosition(vid)
            speed = traci.vehicle.getSpeed(vid)
            vtype = traci.vehicle.getTypeID(vid)
            is_icv = (vtype == "ICV")
            
            # Simple TTC-based Kinematic Risk Calculation
            # To simulate, we just use random high risk for leading vehicles, low for others
            # In a full simulation you'd look at traci.vehicle.getLeader()
            leader = traci.vehicle.getLeader(vid, 100)
            risk_score = 0.1
            if leader is not None:
                ldist = leader[1]
                if ldist < 15 and speed > 15:
                    risk_score = min(1.0, 0.5 + (15-ldist)/30.0 + speed/100.0)
            
            # Update state tracking
            if vid not in vehicles_state:
                vehicles_state[vid] = {'base_aoi': 0.1, 'ours_aoi': 0.1}
                
            vehicles_state[vid]['base_aoi'] += 0.1
            vehicles_state[vid]['ours_aoi'] += 0.1
            
            v_data = {
                'pos': pos,
                'speed': speed,
                'is_icv': is_icv,
                'risk_score': risk_score,
                'aoi': vehicles_state[vid]['ours_aoi'] # Use ours for matching decision
            }
            
            current_vehs[vid] = v_data
            if is_icv:
                icvs[vid] = v_data
            else:
                nicvs[vid] = v_data

        if not icvs or not nicvs:
            continue

        # 1. Matching
        # Base (Risk-blind) - approximating hungarian with risk_aware=False
        base_assignment = greedy_matching(icvs, nicvs, risk_aware=False)
        # Ours (Risk-aware)
        ours_assignment = greedy_matching(icvs, nicvs, risk_aware=True)
        
        # 2. Allocation
        base_alloc = sca_allocation(current_vehs, total_bw)
        ours_alloc = waterfill_allocation(current_vehs, total_bw, risk_aware=True)
        
        # 3. Update AoIs and Log Metrics
        danger_count = 0
        b_bw_d = 0
        o_bw_d = 0
        
        avg_base_aoi = 0
        avg_ours_aoi = 0
        
        for vid, v_data in current_vehs.items():
            # Update Base AoI
            if base_alloc.get(vid, 0) > 0.5: # arbitrary threshold for successful update
                vehicles_state[vid]['base_aoi'] = 0.1
                
            # Update Ours AoI
            if ours_alloc.get(vid, 0) > 0.5:
                vehicles_state[vid]['ours_aoi'] = 0.1
                
            avg_base_aoi += vehicles_state[vid]['base_aoi']
            avg_ours_aoi += vehicles_state[vid]['ours_aoi']
            
            # Track dangerous vehicles
            if v_data['risk_score'] > 0.6:
                danger_count += 1
                b_bw_d += base_alloc.get(vid, 0)
                o_bw_d += ours_alloc.get(vid, 0)
                
        if danger_count > 0:
            base_bw_danger.append(b_bw_d / danger_count)
            ours_bw_danger.append(o_bw_d / danger_count)
            
        base_aoi_history.append(avg_base_aoi / len(current_vehs))
        ours_aoi_history.append(avg_ours_aoi / len(current_vehs))
        
        step += 1
        if step % 100 == 0:
            print(f"Sim Time: {traci.simulation.getTime():.1f}s | Active Vehicles: {len(current_vehs)}")

    traci.close()
    
    print("\n" + "="*60)
    print("SUMO SIMULATION RESULTS (Real Mobility Traces)")
    print("="*60)
    print(f"Average AoI (Base Paper): {np.mean(base_aoi_history):.3f} s")
    print(f"Average AoI (Our Method): {np.mean(ours_aoi_history):.3f} s")
    print(f"AoI Reduction: {(1 - np.mean(ours_aoi_history)/np.mean(base_aoi_history))*100:.1f}%")
    print("-" * 60)
    if base_bw_danger:
        print(f"Avg Bandwidth to High-Risk Vehicles (Base): {np.mean(base_bw_danger):.2f} MHz")
        print(f"Avg Bandwidth to High-Risk Vehicles (Ours): {np.mean(ours_bw_danger):.2f} MHz")
        print(f"Improvement for Dangerous Vehicles: {np.mean(ours_bw_danger)/np.mean(base_bw_danger):.2f}x")
    else:
        print("No dangerous situations occurred in this random seed.")
    print("="*60)

if __name__ == "__main__":
    main()
