import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt


def haversine(lon1, lat1, lon2, lat2):
    """Return great-circle distance in kilometers for two longitude-latitude points."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    r = 6371

    return c * r


def convert_degree_to_degree_minute(decimal_degree):
    """Convert decimal degrees to integer degree and minute components."""
    degree = int(decimal_degree)
    minute = (decimal_degree - degree) * 60
    return degree, minute


def find_nearest_station_distance(event_lat, event_lon, stations_df):
    """Return the nearest station distance from an event epicenter in kilometers."""
    min_distance = float('inf')

    for _, station in stations_df.iterrows():
        station_lat = station['lat']
        station_lon = station['lon']

        distance = haversine(event_lon, event_lat, station_lon, station_lat)

        if distance < min_distance:
            min_distance = distance

    return min_distance


def main():
    """Convert raw GDMS catalog data into the processed final catalog format."""
    print("Reading GDMScatalog.csv...")
    catalog_df = pd.read_csv("../../data/raw/GDMScatalog.csv")

    print("Sorting catalog by time...")
    catalog_df['datetime'] = pd.to_datetime(catalog_df['date'] + ' ' + catalog_df['time'])
    catalog_df = catalog_df.sort_values('datetime').reset_index(drop=True)
    catalog_df = catalog_df.drop('datetime', axis=1)

    print("Reading GDMSstations.csv...")
    stations_df = pd.read_csv("../../data/raw/GDMSstations.csv", header=None,
                              names=['network', 'station', 'lat', 'lon', 'elevation'])

    output_data = []

    print(f"Processing {len(catalog_df)} earthquake events...")

    for event_index, event_row in catalog_df.iterrows():
        date_str = str(event_row['date'])
        time_str = str(event_row['time'])

        year = int(date_str.split('-')[0])
        month = int(date_str.split('-')[1])
        day = int(date_str.split('-')[2])

        hour = int(time_str.split(':')[0])
        minute = int(time_str.split(':')[1])
        second = float(time_str.split(':')[2])

        lat = event_row['lat']
        lon = event_row['lon']

        lat_degree, lat_minute = convert_degree_to_degree_minute(lat)
        lon_degree, lon_minute = convert_degree_to_degree_minute(lon)

        depth = event_row['depth']
        magnitude = event_row['ML']

        nearest_dist = find_nearest_station_distance(lat, lon, stations_df)

        output_record = {
            'EQ_ID': event_index + 1,
            'year': year,
            'month': month,
            'day': day,
            'hour': hour,
            'minute': minute,
            'second': second,
            'lat': lat_degree,
            'lat_minute': round(lat_minute, 2),
            'lon': lon_degree,
            'lon_minute': round(lon_minute, 2),
            'depth': depth,
            'magnitude': magnitude,
            'nearest_sta_dist (km)': round(nearest_dist, 1)
        }

        output_data.append(output_record)

        if (event_index + 1) % 10 == 0:
            print(f"Processed {event_index + 1}/{len(catalog_df)} events...")

    output_df = pd.DataFrame(output_data)

    output_path = "../../data/processed/2025_final_catalog_demo.csv"
    output_df.to_csv(output_path, index=False)

    print(f"\nConversion completed!")
    print(f"Output saved to: {output_path}")
    print(f"Total events processed: {len(output_df)}")
    print("\nFirst few rows of output:")
    print(output_df.head())


if __name__ == "__main__":
    main()