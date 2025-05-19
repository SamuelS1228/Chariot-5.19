
import numpy as np
from sklearn.cluster import KMeans
from utils import warehousing_cost, get_drive_time_matrix

# -------------------------------------------------------------------------
DISTANCE_FACTOR = 1.3  # Factor to approximate road mileage from great‑circle miles

def _miles_from_haversine(lon1, lat1, lon2, lat2):
    """Return great‑circle distance in miles, scaled by DISTANCE_FACTOR.

    Supports scalar or numpy array inputs.
    """
    rad = np.pi / 180.0
    lon1 = np.asarray(lon1)
    lat1 = np.asarray(lat1)
    lon2 = np.asarray(lon2)
    lat2 = np.asarray(lat2)
    dlon = (lon2 - lon1) * rad
    dlat = (lat2 - lat1) * rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2.0) ** 2
    miles = 3958.8 * 2 * np.arcsin(np.sqrt(a))
    return miles * DISTANCE_FACTOR


def _drive_distance_matrix(orig, dest, api_key):
    """Return a matrix of driving distances in **miles** via OpenRouteService.

    Falls back to None (caller should handle fallback) if request fails.
    """
    if not api_key or not orig or not dest:
        return None
    try:
        import openrouteservice
        client = openrouteservice.Client(key=api_key)
        matrix = client.distance_matrix(
            locations=orig + dest,
            profile="driving-car",
            metrics=["distance"],
            sources=list(range(len(orig))),
            destinations=list(range(len(orig), len(orig) + len(dest))),
        )
        # Returned distances are in metres – convert to miles
        return np.array(matrix["distances"]) / 1609.34
    except Exception:
        return None


def _assign(df, centers, api_key):
    """Assign each store to the nearest center (in miles); returns (index, dist_miles)."""
    s_lon = df['Longitude'].values
    s_lat = df['Latitude'].values
    mat = _drive_distance_matrix(np.column_stack([s_lon, s_lat]).tolist(),
                                 centers, api_key)
    if mat is None:
        dists = np.empty((len(df), len(centers)))
        for j, (lon, lat) in enumerate(centers):
            dists[:, j] = _miles_from_haversine(s_lon, s_lat, lon, lat)
    else:
        dists = mat
    idx = dists.argmin(axis=1)
    dist_miles = dists[np.arange(len(df)), idx]
    return idx, dist_miles


def _compute_outbound(df, centers, rate_out_mile, api_key):
    idx, dist_miles = _assign(df, centers, api_key)
    outbound_cost = (df['DemandLbs'] * dist_miles * rate_out_mile).sum()
    return outbound_cost, idx, dist_miles


def _greedy_candidate_select(df, k, fixed, sites, rate_out_mile, api_key):
    selected = fixed.copy()
    remaining = [s for s in sites if s not in selected]
    while len(selected) < k and remaining:
        best_site, best_cost = None, None
        for cand in remaining:
            test = selected + [cand]
            cost = _compute_outbound(df, test, rate_out_mile, api_key)[0]
            if best_cost is None or cost < best_cost:
                best_cost, best_site = cost, cand
        selected.append(best_site)
        remaining.remove(best_site)
    return selected


def optimize(
    df,
    k_vals,
    rate_out_mile,
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
    use_drive_times=False,
    ors_api_key=None,
    candidate_sites=None,
    restrict_cand=False,
    candidate_costs=None,
):
    """Core optimization routine (unchanged except cost now per lb‑mile)."""
    inbound_pts = inbound_pts or []
    fixed_centers = fixed_centers or []
    rdc_list = rdc_list or []
    candidate_costs = candidate_costs or {}
    best = None

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
                df, k_eff, fixed_centers, candidate_sites,
                rate_out_mile, ors_api_key if use_drive_times else None
            )
        else:
            km = KMeans(n_clusters=k_eff, n_init=10, random_state=42).fit(df[['Longitude', 'Latitude']])
            centers = km.cluster_centers_.tolist()
            # override with fixed centers
            for i_fc, fc in enumerate(fixed_centers):
                centers[i_fc] = fc

        # assignment of stores to warehouses
        idx, dist_miles = _assign(df, centers, ors_api_key if use_drive_times else None)
        assigned = df.copy()
        assigned['Warehouse'] = idx
        assigned['DistMiles'] = dist_miles

        # outbound cost
        out_cost = (assigned['DemandLbs'] * dist_miles * rate_out_mile).sum()

        # warehousing cost
        demand_per_wh = []
        wh_cost = 0.0
        for i, (lon, lat) in enumerate(centers):
            dem = assigned.loc[assigned['Warehouse'] == i, 'DemandLbs'].sum()
            demand_per_wh.append(dem)
            used_cost_sqft = _cost_for_center(lon, lat)
            wh_cost += warehousing_cost(dem, sqft_per_lb, used_cost_sqft, fixed_cost)

        # inbound cost
        in_cost = 0.0
        if consider_inbound and inbound_pts:
            c_coords = centers
            for lon, lat, pct in inbound_pts:
                mat = _drive_distance_matrix([[lon, lat]], c_coords,
                                             ors_api_key if use_drive_times else None)
                if mat is None:
                    dists = [_miles_from_haversine(lon, lat, cx, cy) for cx, cy in c_coords]
                else:
                    dists = mat[0]
                in_cost += (np.array(dists) * np.array(demand_per_wh) * pct * inbound_rate_mile).sum()

        # transfer cost (RDC to WH)
        trans_cost = 0.0
        rdc_only = [r for r in rdc_list if not r['is_sdc']]
        if rdc_only:
            wh_coords = centers
            r_coords = [r['coords'] for r in rdc_only]
            mat = _drive_distance_matrix(r_coords, wh_coords,
                                         ors_api_key if use_drive_times else None)
            if mat is None:
                mat = np.array([[ _miles_from_haversine(rx, ry, wx, wy)
                                  for wx, wy in wh_coords]
                                 for rx, ry in r_coords])
            share = 1.0 / len(r_coords)
            for row in mat:
                trans_cost += (row * np.array(demand_per_wh) * share * transfer_rate_mile).sum()

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
