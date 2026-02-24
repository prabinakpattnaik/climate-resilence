"""
Download ENSO MEI v2 (Multivariate ENSO Index) from NOAA.
ENSO is the #1 driver of Indian monsoon variability.

Source: https://psl.noaa.gov/enso/mei/data/meiv2.data
Format: Year + 12 bimonthly columns (DJ, JF, FM, MA, AM, MJ, JJ, JA, AS, SO, ON, ND)
Positive values = El Nino, Negative = La Nina
"""

import requests
import pandas as pd
import numpy as np
import os
import re


def download_enso_mei():
    """Download ENSO MEI v2 data from NOAA and save as clean CSV."""

    print("=" * 60)
    print("DOWNLOADING ENSO MEI v2 (El Nino / La Nina Index)")
    print("Source: NOAA Physical Sciences Laboratory")
    print("=" * 60)

    url = "https://psl.noaa.gov/enso/mei/data/meiv2.data"

    print(f"\n1. Fetching from {url}...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        raw_text = response.text

        # Parse the space-delimited text data
        lines = raw_text.strip().split("\n")

        # Bimonthly column mapping to center month:
        # DJ=Jan, JF=Feb, FM=Mar, MA=Apr, AM=May, MJ=Jun,
        # JJ=Jul, JA=Aug, AS=Sep, SO=Oct, ON=Nov, ND=Dec
        bimonthly_to_month = {
            0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6,
            6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12
        }

        rows = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            # Try to parse year (first field)
            try:
                year = int(float(parts[0]))
            except (ValueError, IndexError):
                continue

            # Skip header-like lines
            if year < 1900 or year > 2100:
                continue

            # Parse 12 bimonthly values
            for i, val_str in enumerate(parts[1:13]):
                try:
                    val = float(val_str)
                except ValueError:
                    continue

                month = bimonthly_to_month.get(i, i + 1)

                # -999.00 is the missing value sentinel
                if val <= -999.0:
                    continue

                # Sanity check: MEI values are typically between -4 and +4
                if abs(val) > 5.0:
                    continue

                rows.append({
                    "year": year,
                    "month": month,
                    "mei_value": round(val, 3)
                })

        df = pd.DataFrame(rows)
        df = df.sort_values(["year", "month"]).reset_index(drop=True)

        print(f"\n2. Parsed {len(df)} monthly ENSO records")
        print(f"   Year range: {df['year'].min()}-{df['year'].max()}")
        print(f"   MEI range: {df['mei_value'].min():.3f} to {df['mei_value'].max():.3f}")
        print(f"   El Nino months (MEI > 0.5): {(df['mei_value'] > 0.5).sum()}")
        print(f"   La Nina months (MEI < -0.5): {(df['mei_value'] < -0.5).sum()}")
        print(f"   Neutral months: {((df['mei_value'] >= -0.5) & (df['mei_value'] <= 0.5)).sum()}")

        # Save to data folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "..", "data")
        os.makedirs(data_dir, exist_ok=True)

        output_path = os.path.join(data_dir, "enso_mei.csv")
        df.to_csv(output_path, index=False)

        print(f"\n3. Saved to: {output_path}")
        print(f"   Columns: year, month, mei_value")

        # Show recent values for verification
        recent = df[df["year"] >= 2023].tail(12)
        print(f"\n   Recent ENSO values:")
        for _, row in recent.iterrows():
            status = "El Nino" if row["mei_value"] > 0.5 else ("La Nina" if row["mei_value"] < -0.5 else "Neutral")
            print(f"   {int(row['year'])}-{int(row['month']):02d}: MEI={row['mei_value']:+.3f} ({status})")

        print("\n" + "=" * 60)
        print("ENSO DATA DOWNLOAD COMPLETE")
        print("=" * 60)

        return df

    except requests.exceptions.RequestException as e:
        print(f"\nERROR: Failed to fetch data - {e}")
        return None


if __name__ == "__main__":
    download_enso_mei()
