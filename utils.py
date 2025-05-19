
import math
EARTH_RADIUS_MI=3958.8

def haversine(lon1,lat1,lon2,lat2):
    lon1,lat1,lon2,lat2=map(math.radians,[lon1,lat1,lon2,lat2])
    dlon=lon2-lon1
    dlat=lat2-lat1
    a=math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return EARTH_RADIUS_MI*2*math.asin(math.sqrt(a))

def warehousing_cost(demand_lbs,sqft_per_lb,cost_per_sqft,fixed_cost):
    return fixed_cost + demand_lbs*sqft_per_lb*cost_per_sqft
