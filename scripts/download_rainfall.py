"""
Download recent rainfall data (2022-2025) from Open-Meteo API for Hyderabad
and append to the existing hyderabad_rainfall_data.csv.

The existing CSV has data from 1901-2021. This script extends it to 2025.
"""

import requests
import pandas as pd
import numpy as np
import os


def download_and_append_rainfall():
    """Download 2022-2025 monthly rainfall and append to existing CSV."""

    print("=" * 60)
    print("DOWNLOADING HYDERABAD RAINFALL DATA (2022-2025)")
    print("Source: Open-Meteo Historical Weather API (FREE)")
    print("=" * 60)

    # Hyderabad coordinates
    LAT = 17.385
    LON = 78.4867

    # Fetch 2022 to 2025
    START_DATE = "2022-01-01"
    END_DATE = "2025-12-31"

    # Open-Meteo Historical API
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": "precipitation_sum",
        "timezone": "Asia/Kolkata"
    }

    print(f"\n1. Fetching daily precipitation from Open-Meteo API...")
    print(f"   Location: Hyderabad ({LAT}, {LON})")
    print(f"   Period: {START_DATE} to {END_DATE}")

    try:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()

        daily = data["daily"]
        df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "precipitation": daily.get("precipitation_sum", [0] * len(daily["time"]))
        })
        df["precipitation"] = df["precipitation"].fillna(0.0)

        print(f"   Retrieved {len(df)} daily records")

        # Aggregate daily -> monthly totals
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        monthly = df.groupby(["year", "month"])["precipitation"].sum().reset_index()

        print(f"\n2. Aggregated to {len(monthly)} monthly records")

        # Exact column names from existing CSV (note inconsistencies: April, June, July, Aug, Sept)
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "Aug",
            9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec"
        }

        # Pivot to wide format matching existing CSV structure
        rows = []
        for year in sorted(monthly["year"].unique()):
            year_data = monthly[monthly["year"] == year]
            row = {"Year": int(year)}
            total = 0.0
            for m in range(1, 13):
                month_val = year_data[year_data["month"] == m]["precipitation"]
                val = round(float(month_val.values[0]), 2) if len(month_val) > 0 else 0.0
                row[month_names[m]] = val
                total += val
            row["Total"] = round(total, 2)
            rows.append(row)

        new_df = pd.DataFrame(rows)
        print(f"   Pivoted to wide format: {len(new_df)} year rows")
        for _, row in new_df.iterrows():
            print(f"   {int(row['Year'])}: Total = {row['Total']:.1f} mm")

        # Load existing CSV and append
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "..", "data")
        csv_path = os.path.join(data_dir, "hyderabad_rainfall_data.csv")

        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path)
            existing_years = set(existing["Year"].astype(int).values)
            print(f"\n3. Existing CSV: {len(existing)} rows, years {int(existing['Year'].min())}-{int(existing['Year'].max())}")

            # Only append years that don't already exist
            new_years = new_df[~new_df["Year"].isin(existing_years)]
            if len(new_years) == 0:
                print("   No new years to append - CSV is already up to date!")
                return existing

            combined = pd.concat([existing, new_years], ignore_index=True)
            combined = combined.sort_values("Year").reset_index(drop=True)
            print(f"   Appending {len(new_years)} new years")
        else:
            combined = new_df
            print(f"\n3. No existing CSV found, creating new file with {len(new_df)} rows")

        combined.to_csv(csv_path, index=False)
        print(f"\n4. Saved to: {csv_path}")
        print(f"   Total rows: {len(combined)}")
        print(f"   Year range: {int(combined['Year'].min())}-{int(combined['Year'].max())}")
        print("\n" + "=" * 60)
        print("RAINFALL DATA UPDATE COMPLETE")
        print("=" * 60)

        return combined

    except requests.exceptions.RequestException as e:
        print(f"\nERROR: Failed to fetch data - {e}")
        return None


if __name__ == "__main__":
    download_and_append_rainfall()
