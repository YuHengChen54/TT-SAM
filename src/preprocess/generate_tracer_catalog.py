import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt, atan2, degrees


def haversine(lon1, lat1, lon2, lat2):
    """Return great-circle distance in kilometers between two points."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    r = 6371

    return c * r


def calculate_azimuth(lat1, lon1, lat2, lon2):
    """Return azimuth in degrees from the first point to the second point."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1

    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)

    initial_bearing = atan2(x, y)

    bearing = (degrees(initial_bearing) + 360) % 360

    return bearing


def main():
    """Generate trace-level catalog records for all earthquake-station pairs."""
    print("Reading GDMScatalog.csv...")
    catalog_df = pd.read_csv("../../data/raw/GDMScatalog.csv")

    print("Sorting catalog by time...")
    catalog_df['datetime'] = pd.to_datetime(catalog_df['date'] + ' ' + catalog_df['time'])
    catalog_df = catalog_df.sort_values('datetime').reset_index(drop=True)
    catalog_df = catalog_df.drop('datetime', axis=1)

    print("Reading GDMSstations_Vs30.csv...")
    stations_df = pd.read_csv("../../data/processed/GDMSstations_Vs30.csv")

    output_data = []

    print(f"Processing {len(catalog_df)} earthquake events with {len(stations_df)} stations...")
    print(f"Total records to generate: {len(catalog_df) * len(stations_df)}")

    for eq_idx, eq_row in catalog_df.iterrows():
        date_str = str(eq_row['date'])
        time_str = str(eq_row['time'])

        year = int(date_str.split('-')[0])
        month = int(date_str.split('-')[1])
        day = int(date_str.split('-')[2])

        hour = int(time_str.split(':')[0])
        minute = int(time_str.split(':')[1])
        second = float(time_str.split(':')[2])

        eq_lat = eq_row['lat']
        eq_lon = eq_row['lon']
        eq_depth = eq_row['depth']
        eq_magnitude = eq_row['ML']

        for _, station_row in stations_df.iterrows():
            station_info = station_row['station_info']
            if '(' in station_info and ')' in station_info:
                station_code = station_info.split('(')[0]
                station_name = station_info.split('(')[1].split(')')[0]
            else:
                station_code = station_info
                station_name = station_info
            station_lat = station_row['latitude']
            station_lon = station_row['longitude']
            station_elevation = station_row['elevation']
            station_vs30 = station_row['Vs30']

            epdis = haversine(eq_lon, eq_lat, station_lon, station_lat)

            sta_angle = calculate_azimuth(eq_lat, eq_lon, station_lat, station_lon)

            output_record = {
                'EQ_ID': eq_idx + 1,
                'year': year,
                'month': month,
                'day': day,
                'hour': hour,
                'minute': minute,
                'second': second,
                'station_code': station_code,
                'station_name': station_name,
                'epdis (km)': round(epdis, 2),
                'latitude': eq_lat,
                'longitude': eq_lon,
                'elevation': station_elevation,
                'sta_angle': round(sta_angle, 0),
                'depth': eq_depth,
                'magnitude': eq_magnitude,
                'Vs30': station_vs30
            }

            output_data.append(output_record)

        if (eq_idx + 1) % 1 == 0:
            print(f"Processed {eq_idx + 1}/{len(catalog_df)} events ({len(output_data)} total records)...")

    output_df = pd.DataFrame(output_data)

    output_path = "../../data/processed/2025_final_traces_demo.csv"
    output_df.to_csv(output_path, index=False)

    print(f"\nConversion completed!")
    print(f"Output saved to: {output_path}")
    print(f"Total records generated: {len(output_df)}")
    print(f"Earthquakes: {len(catalog_df)}, Stations: {len(stations_df)}")
    print("\nFirst few rows of output:")
    print(output_df.head(10))
    print("\nColumn names:")
    print(output_df.columns.tolist())


if __name__ == "__main__":
    main()