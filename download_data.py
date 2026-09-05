from pathlib import Path
from urllib.request import urlopen, Request

DATA_URL = "https://raw.githubusercontent.com/amankharwal/Website-data/master/marketing_campaign.csv"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_PATH = DATA_DIR / "marketing_campaign.csv"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        print(f"Dataset already exists: {DATA_PATH}")
        return

    print("Downloading public Customer Personality Analysis dataset...")
    try:
        request = Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            content = response.read()
        DATA_PATH.write_bytes(content)
        print(f"Saved {len(content):,} bytes to {DATA_PATH}")
    except Exception as exc:
        print("Automatic download failed.")
        print("Download the dataset from:")
        print("https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis")
        print(f"Then place marketing_campaign.csv in: {DATA_DIR}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
