import numpy as np

def centroid_X_Waterplane(full_breadths, station_X):
    area_stations = np.zeros(np.size(full_breadths), dtype= np.float64)
    centroid_stations_X = np.zeros(np.size(full_breadths), dtype= np.float64)

    print(area_stations, centroid_stations_X)

    for i in range(np.size(full_breadths) - 1):
        print(i)
        area_stations[i] = ((full_breadths[i] + full_breadths[i + 1]) * (station_X[ i + 1] - station_X[i]) * 0.5)  #area of trapezium formula
        centroid_stations_X[i] = (station_X[i] + station_X[i + 1]) / 2

    total_area_waterplane = np.sum(area_stations) 
    area_centroid_each_station = area_stations * centroid_stations_X

    centroid_water_plane = np.sum(area_centroid_each_station) / total_area_waterplane

    return centroid_water_plane, total_area_waterplane

def Second_order_inertia(Centroid, half_breadths, stations_X):
    moment_X = 0.0
    moment_Y = 0.0

    for i in range(np.size(half_breadths) - 1):
        moment_Y += (half_breadths[i + 1] + half_breadths[i]) * (stations_X[i + 1] - stations_X[i]) * ((((station_X[i] + station_X[i + 1]) / 2) - Centroid)**2)
        moment_X += (((half_breadths[i + 1] + half_breadths[i]) / 4)**2) * ((stations_X[i + 1] - stations_X[i]) * (half_breadths[i + 1] + half_breadths[i]))

    return moment_X, moment_Y

length_of_ship = 80
stations = np.array([1, 2, 3, 4, 5, 6, 7, 7.5, 8], dtype= np.float64)
half_breadths = np.array([0.0, 3, 6, 9, 10, 10, 8.5, 6, 0.0], dtype= np.float64)

print(stations, half_breadths)

full_breadths = half_breadths * 2

print(full_breadths)
station_X = stations * length_of_ship

centroid_water_plane, total_water_plane_area = centroid_X_Waterplane(full_breadths, station_X)

moment_X, moment_Y = Second_order_inertia(centroid_water_plane, half_breadths, station_X)

print(f"total_water_plae_area = {total_water_plane_area}, centroid = {centroid_water_plane} moment_X = {moment_X}, moment_Y= {moment_Y}")



