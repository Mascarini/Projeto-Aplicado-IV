import os
import time
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import requests
from tqdm import tqdm

# Load .env from project root (parent of dataset/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_BASE_URL = "https://api.mosqlimate.org/api/datastore/infodengue/"
API_KEY = os.getenv("MOSQLIMATE_API_KEY")
PER_PAGE = 300


def create_dataset_folder() -> Path:
    folder = Path(__file__).resolve().parent
    folder.mkdir(exist_ok=True)
    return folder


def fetch_infodengue_data(
    disease: str = "dengue",
    uf: str = "SP",
    start_date: str = "2015-01-01",
    end_date: str = "2026-03-01",
    max_retries: int = 3
) -> pd.DataFrame:
    
    if not API_KEY:
        raise ValueError("MOSQLIMATE_API_KEY not found in .env file")
    
    all_data = []
    page = 1
    total_pages = None
    pbar = None
    
    headers = {"X-UID-Key": API_KEY}
    
    while True:
        params = {
            "page": page,
            "per_page": PER_PAGE,
            "disease": disease,
            "start": start_date,
            "end": end_date,
            "uf": uf
        }
        
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(API_BASE_URL, params=params, headers=headers, timeout=60)
                response.raise_for_status()
                data = response.json()
                
                if total_pages is None:
                    pagination = data.get("pagination", {})
                    total_pages = pagination.get("total_pages", 1)
                    total_items = pagination.get("total_items", 0)
                    print(f"Disease: {disease.upper()} | State: {uf}")
                    print(f"Period: {start_date} to {end_date}")
                    print(f"Total records: {total_items:,} | Pages: {total_pages}")
                    pbar = tqdm(total=total_pages, desc="Downloading", unit="page")
                
                items = data.get("items", [])
                if not items:
                    break
                
                all_data.extend(items)
                
                if pbar:
                    pbar.update(1)
                
                break
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    time.sleep(2 * attempt)
                else:
                    print(f"Failed after {max_retries} attempts on page {page}: {e}")
                    if pbar:
                        pbar.close()
                    return pd.DataFrame(all_data)
        
        if page >= total_pages:
            break
        
        page += 1
        time.sleep(0.2)
    
    if pbar:
        pbar.close()
    
    return pd.DataFrame(all_data)


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    column_mapping = {
        "data_iniSE": "week_start_date",
        "SE": "epi_week",
        "casos_est": "estimated_cases",
        "casos_est_min": "estimated_cases_min",
        "casos_est_max": "estimated_cases_max",
        "casos": "notified_cases",
        "municipio_geocodigo": "ibge_code",
        "municipio_nome": "municipality",
        "p_rt1": "prob_rt_above_1",
        "p_inc100k": "incidence_100k",
        "nivel": "alert_level",
        "Rt": "reproduction_number",
        "pop": "population",
        "receptivo": "climate_receptivity",
        "transmissao": "transmission_evidence",
        "nivel_inc": "incidence_level",
        "umidmax": "humidity_max",
        "umidmed": "humidity_avg",
        "umidmin": "humidity_min",
        "tempmax": "temperature_max",
        "tempmed": "temperature_avg",
        "tempmin": "temperature_min",
        "casprov": "probable_cases",
        "casprov_est": "probable_cases_est",
        "casprov_est_min": "probable_cases_est_min",
        "casprov_est_max": "probable_cases_est_max",
        "casconf": "lab_confirmed_cases"
    }
    
    df = df.rename(columns=column_mapping)
    
    if "week_start_date" in df.columns:
        df["week_start_date"] = pd.to_datetime(df["week_start_date"], errors="coerce")
    
    sort_columns = [col for col in ["ibge_code", "week_start_date"] if col in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns).reset_index(drop=True)
    
    return df


def aggregate_by_state(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    sum_columns = ["notified_cases", "estimated_cases", "probable_cases", "lab_confirmed_cases", "population"]
    avg_columns = ["temperature_avg", "temperature_max", "temperature_min", "humidity_avg", "humidity_max", "humidity_min"]
    
    agg_dict = {}
    for col in sum_columns:
        if col in df.columns:
            agg_dict[col] = "sum"
    for col in avg_columns:
        if col in df.columns:
            agg_dict[col] = "mean"
    
    if "week_start_date" not in df.columns:
        return df
    
    df_state = df.groupby(["epi_week", "week_start_date"]).agg(agg_dict).reset_index()
    
    if "notified_cases" in df_state.columns and "population" in df_state.columns:
        df_state["incidence_100k"] = (df_state["notified_cases"] / df_state["population"] * 100000).round(2)
    
    return df_state.sort_values("week_start_date").reset_index(drop=True)


def main():
    print("=" * 70)
    print("INFODENGUE DATA EXTRACTION - SAO PAULO")
    print("Source: Mosqlimate API (info.dengue.mat.br)")
    print("=" * 70)
    
    dataset_folder = create_dataset_folder()
    
    CONFIG = {
        "uf": "SP",
        "start_date": "2015-01-01",
        "end_date": "2026-12-31"
    }
    
    diseases = ["dengue"]
    
    for disease in diseases:
        print(f"\n{'=' * 70}")
        print(f"Fetching {disease.upper()} data...")
        print("=" * 70)
        
        df = fetch_infodengue_data(
            disease=disease,
            uf=CONFIG["uf"],
            start_date=CONFIG["start_date"],
            end_date=CONFIG["end_date"]
        )
        
        if df.empty:
            print(f"No data retrieved for {disease}")
            continue
        
        df = process_dataframe(df)
        
        municipalities_file = dataset_folder / f"infodengue_{disease}_sp_municipalities.csv"
        df.to_csv(municipalities_file, index=False)
        print(f"\nSaved: {municipalities_file}")
        print(f"  Records: {len(df):,}")
        print(f"  Municipalities: {df['ibge_code'].nunique() if 'ibge_code' in df.columns else 'N/A'}")
        
        if disease == "dengue":
            df_state = aggregate_by_state(df)
            state_file = dataset_folder / "infodengue_dengue_sp_state.csv"
            df_state.to_csv(state_file, index=False)
            print(f"\nSaved: {state_file}")
            print(f"  Epidemiological weeks: {len(df_state):,}")
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    
    print("\nGenerated files:")
    for file in dataset_folder.glob("infodengue_*.csv"):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
