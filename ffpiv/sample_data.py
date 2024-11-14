"""Retrieval of sample dataset."""

import glob
import os.path
import zipfile


def get_hommerich_dataset():
    """Retrieve and cache sample dataset of Sheaf river."""
    try:
        import pooch
    except ImportError:
        raise ImportError("This function needs pooch. Install ffpiv with pip install ffpiv[extra]")

    # Define the DOI link
    filename = "hommerich_frames_20241010_081717.zip"
    base_url = "doi:10.5281/zenodo.14161026"
    url = base_url + "/" + filename
    print(f"Retrieving or providing cached version of dataset from {url}")
    # Create a Pooch registry to manage downloads
    registry = pooch.create(
        # Define the cache directory
        path=pooch.os_cache("ffpiv"),
        # Define the base URL for fetching data
        base_url=base_url,
        # Define the registry with the file we're expecting to download
        registry={filename: None},
    )
    # Fetch the dataset
    file_path = registry.fetch(filename, progressbar=True)
    print(f"Hommerich dataset is available in {file_path}")
    return file_path


def get_hommerich_files():
    """Unzip hommerich dataset and return file list."""
    zip_file = get_hommerich_dataset()
    trg_dir = os.path.split(zip_file)[0]
    trg_subdir = os.path.join(trg_dir, "hommerich_frames_20241010_081717")
    jpg_template = os.path.join(trg_subdir, "frame*.jpg")
    if not (len(glob.glob(jpg_template)) >= 122):
        # apparently the jpg files are incomplete or not present. Unzip the record.
        print("Unzipping sample data...")
        with zipfile.ZipFile(zip_file, "r") as f:
            f.extractall(trg_dir)
    # make a neat sorted list of jpg files
    jpg_files = glob.glob(os.path.join(trg_dir, "hommerich_frames_20241010_081717", "frame*.jpg"))
    jpg_files.sort()
    return jpg_files
