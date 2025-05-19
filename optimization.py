
import numpy as np
from sklearn.cluster import KMeans
from utils import haversine, warehousing_cost

# -------------------------------------------------------------------------
def _distance_matrix(orig_lon, orig_lat, centers):
    """Return great‑circle distance matrix in miles."""
    dists = np.empty((len(orig_lon), len(centers)))
    for j, (clon, clat) in enumerate(centers):
        dists[:, j] = haversine(orig_lon, orig_lat, clon, clat)
    return dists

def _assign(df, centers):
    s_lon = df['Longitude'].values
    s_lat = df['Latitude'].values
    dists = _distance_matrix(s_lon, s_lat, centers)
    idx = dists.argmin(axis=1)
    dmin = dists[np.arange(len(df)), idx]
    return idx, dmin

# -------------------------------------------------------------------------
def _greedy_candidate_select(df, k, fixed, sites, rate_out):
    selected = fixed.copy()
    remaining = [s for s in sites if s not in selected]
    while len(selected) < k and remaining:
        best_site, best_cost = None, None
        for cand in remaining:
            test = selected + [cand]
            cost = _compute_outbound(df, test, rate_out)[0]
            if best_cost is None or cost < best_cost:
                best_cost, best_site = cost, cand
        selected.append(best_site)
        remaining.remove(best_site)
    return selected

# -------------------------------------------------------------------------
def _compute_outbound(df, centers, rate_out):
    idx, dmin = _assign(df, centers)
    outbound_cost = (df['DemandLbs'] * dmin * rate_out).sum()
    return outbound_cost, idx, dmin

# -------------------------------------------------------------------------
def optimize(
    df,
    k_vals,
    rate_out,
    sqft_per_lb,
    cost_sqft,
    fixed_cost,
    consider_inbound=False,
    inbound_rate_mile=0.0,
    inbound_pts=None,
    fixed_centers=None,
    rdc_list=None,
    transfer_rate_mile=0.0,
    rdc_sqft_per_lb=None,
    rdc_cost_per_sqft=None,
    candidate_sites=None,
    restrict_cand=False,
    candidate_costs=None,
):
    inbound_pts = inbound_pts or []
    fixed_centers = fixed_centers or []
    rdc_list = rdc_list or []
    candidate_costs = candidate_costs or {}
    best = None

    # helper to fetch cost per sqft for a given center
    def _cost_for_center(lon, lat):
        if restrict_cand:
            key = (round(float(lon), 6), round(float(lat), 6))
            return candidate_costs.get(key, cost_sqft)
        return cost_sqft

    for k in k_vals:
        k_eff = max(k, len(fixed_centers))

        # ----- choose center locations -----------------------------------
        if candidate_sites and len(candidate_sites) >= k_eff:
            centers = _greedy_candidate_select(
                df, k_eff, fixed_centers, candidate_sites, rate_out
            )
        else:
            km = KMeans(n_clusters=k_eff, n_init=10, random_state=42).fit(df[['Longitude', 'Latitude']])
            centers = km.cluster_centers_.tolist()
            # override with fixed centers
            for i_fc, fc in enumerate(fixed_centers):
                centers[i_fc] = fc

        # assignment
        idx, dmin = _assign(df, centers)
        assigned = df.copy()
        assigned['Warehouse'] = idx
        assigned['DistanceMi'] = dmin

        # outbound
        out_cost = (assigned['DemandLbs'] * dmin * rate_out).sum()

        # warehousing
        demand_per_wh = []
        wh_cost = 0.0
        for i, (lon, lat) in enumerate(centers):
            dem = assigned.loc[assigned['Warehouse'] == i, 'DemandLbs'].sum()
            demand_per_wh.append(dem)
            used_cost_sqft = _cost_for_center(lon, lat)
            wh_cost += warehousing_cost(dem, sqft_per_lb, used_cost_sqft, fixed_cost)

        # inbound
        in_cost = 0.0
        if consider_inbound and inbound_pts:
            c_coords = centers
            for lon, lat, pct in inbound_pts:
                dists = np.array([haversine(lon, lat, cx, cy) for cx, cy in c_coords])
                in_cost += (dists * np.array(demand_per_wh) * pct * inbound_rate_mile).sum()

        # transfer (simple model)
        trans_cost = 0.0
        rdc_only = [r for r in rdc_list if not r['is_sdc']]
        if rdc_only:
            wh_coords = centers
            r_coords = [r['coords'] for r in rdc_only]
            share = 1.0 / len(r_coords)
            for rx, ry in r_coords:
                dists = np.array([haversine(rx, ry, wx, wy) for wx, wy in wh_coords])
                trans_cost += (dists * np.array(demand_per_wh) * share * transfer_rate_mile).sum()

        total_cost = out_cost + wh_cost + in_cost + trans_cost

        if best is None or total_cost < best['total_cost']:
            best = dict(
                centers=centers,
                assigned=assigned,
                demand_per_wh=demand_per_wh,
                total_cost=total_cost,
                out_cost=out_cost,
                in_cost=in_cost,
                trans_cost=trans_cost,
                wh_cost=wh_cost,
                rdc_list=rdc_list,
            )
    return best
