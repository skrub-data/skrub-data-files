import argparse
import datetime
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
from skrub import datasets


def create_archive(
    all_datasets_dir, all_archives_dir, dataset_name, dataframes, metadata
):
    print(dataset_name)
    dataset_dir = all_datasets_dir / dataset_name
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata), "utf-8")
    for stem, df in dataframes.items():
        csv_path = dataset_dir / f"{stem}.csv"
        df.to_csv(csv_path, index=False)
    archive_path = all_archives_dir / dataset_name
    result = shutil.make_archive(
        archive_path,
        "zip",
        root_dir=all_datasets_dir,
        base_dir=dataset_name,
    )
    result = Path(result)
    checksum = hashlib.sha256(result.read_bytes()).hexdigest()
    return checksum


def get_metadata(dataset, name):
    result = {}
    result["name"] = getattr(dataset, "name", name)
    for key in ["description", "source", "target"]:
        if value := getattr(dataset, key, None):
            result[key] = value
    return result


def load_simple_dataset(fetcher):
    dataset = fetcher()
    df = dataset.X
    df[dataset.target] = dataset.y
    name = fetcher.__name__.removeprefix("fetch_")
    return (name, {name: df}, get_metadata(dataset, name))


def _world_bank():
    result = {}
    df = pd.read_csv(
        (
            "https://raw.githubusercontent.com/skrub-data/datasets/"
            "master/data/Happiness_report_2022.csv"
        ),
        thousands=",",
    )
    df.drop(df.tail(1).index, inplace=True)
    result["happiness_report"] = df
    result["GDP_per_capita"] = datasets.fetch_world_bank_indicator(
        indicator_id="NY.GDP.PCAP.CD"
    ).X
    result["life_expectancy"] = datasets.fetch_world_bank_indicator("SP.DYN.LE00.IN").X
    result["legal_rights_index"] = datasets.fetch_world_bank_indicator(
        "IC.LGL.CRED.XQ"
    ).X
    description = (
        "Happiness score and relevant country data from the World Bank API. "
        "The table 'happiness_report' comes from the 2022 World Happiness Report "
        "worldhappiness.report, all other tables come from the World Bank "
        "open data platform worldbank.org"
    )
    metadata = {
        "name": "happiness_score",
        "description": description,
        "source": [
            "https://api.worldbank.org/v2/",
            "https://worldhappiness.report/",
        ],
    }
    return ("country_happiness", result, metadata)


def _movielens():
    ratings = datasets.fetch_movielens("ratings").X
    movies_dataset = datasets.fetch_movielens("movies")
    movies = movies_dataset.X
    result = {"movies": movies, "ratings": ratings}
    metadata = get_metadata(movies_dataset, "movielens")
    return "movielens", result, metadata


def _bikes():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/skrub-data/datasets/master"
        "/data/bike-sharing-dataset.csv"
    )
    return (
        "bike_sharing",
        {"bike_sharing": data},
        {"name": "bike_sharing", "target": "cnt"},
    )


def _vg_sales():
    url = (
        "https://raw.githubusercontent.com/William2064888/vgsales.csv/main/vgsales.csv"
    )
    X = pd.read_csv(
        url,
        sep=";",
        on_bad_lines="skip",
    )
    return (
        "videogame_sales",
        {"videogame_sales": X},
        {"name": "videogame_sales", "source": url, "target": "Global_Sales"},
    )


def _flights():
    flights = datasets.fetch_figshare("41771418").X
    airports = datasets.fetch_figshare("41710257").X
    weather = datasets.fetch_figshare("41771457").X
    stations = datasets.fetch_figshare("41710524").X
    return (
        "flight_delays",
        {
            "flights": flights,
            "airports": airports,
            "weather": weather,
            "stations": stations,
        },
        {"name": "flight_delays"},
    )


def _fraud():
    fraud = datasets.fetch_credit_fraud()
    return (
        "credit_fraud",
        {"baskets": fraud.baskets, "products": fraud.products},
        get_metadata(fraud, "credit_fraud"),
    )


def iter_datasets():
    simple_fetchers = {f for f in datasets.__all__ if f.startswith("fetch_")} - {
        "fetch_movielens",
        "fetch_world_bank_indicator",
        "fetch_figshare",
        "fetch_credit_fraud",
        "fetch_ken_embeddings",
        "fetch_ken_table_aliases",
        "fetch_ken_types",
    }
    for fetcher in sorted(simple_fetchers):
        yield load_simple_dataset(getattr(datasets, fetcher))
    yield _fraud()
    yield _world_bank()
    yield _movielens()
    yield _bikes()
    yield _vg_sales()
    yield _flights()


def make_skrub_datasets():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        help="where to store the output. a subdirectory containing all the archives will be created",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(args.output_dir).resolve()

    root_dir = (
        output_dir / f"skrub_datasets_{datetime.datetime.now():%Y-%m-%dT%H-%M%S}"
    )
    root_dir.mkdir(parents=True)
    all_datasets_dir = root_dir / "datasets"
    all_datasets_dir.mkdir()
    all_archives_dir = root_dir / "archives"
    all_archives_dir.mkdir()

    print(f"saving output in {root_dir}")

    checksums = {}
    for dataset_name, dataframes, metadata in iter_datasets():
        if len(dataframes) > 1:
            metadata.pop("target", None)
        checksums[dataset_name] = create_archive(
            all_datasets_dir, all_archives_dir, dataset_name, dataframes, metadata
        )

    (all_archives_dir / "sha256_checksums.json").write_text(
        json.dumps(checksums), "utf-8"
    )
    print(f"archive files saved in {all_archives_dir}")


if __name__ == "__main__":
    make_skrub_datasets()
