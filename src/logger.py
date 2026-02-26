"""
Experiment logger — writes per-seed CSVs to an output folder.

File naming
-----------
Each CSV is saved as:
    <output_dir>/<env_label>_seed<seed>.csv

where <env_label> comes from CoordinationEnv.label and encodes the full
hypothesis configuration (partner, distance, frame, coupling + any dataclass params).

Usage
-----
    logger = RunLogger("outputs")
    env.run_seeds(n_trials=300, n_seeds=10, logger=logger)

    # offline
    df = RunLogger.load("outputs/FixedPartner__SymmetricDistance__…_seed0.csv")
"""

from pathlib import Path

import polars as pl


class RunLogger:
    """Saves each seed's trial-level DataFrame to a separate CSV file."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log(self, seed: int, df: pl.DataFrame, label: str) -> Path:
        """
        Write one seed's DataFrame to CSV.

        Parameters
        ----------
        seed  : seed index (used in filename)
        df    : polars DataFrame from CoordinationEnv._to_dataframe
        label : CoordinationEnv.label string (used as filename prefix)

        Returns
        -------
        Path of the written file.
        """
        path = self.output_dir / f"{label}_seed{seed}.csv"
        df.write_csv(path)
        return path

    @staticmethod
    def load(path: str | Path) -> pl.DataFrame:
        """Load a single seed CSV."""
        return pl.read_csv(path)

    @staticmethod
    def load_all(output_dir: str | Path, label: str) -> pl.DataFrame:
        """
        Load and concatenate all seed CSVs for a given label.

        Parameters
        ----------
        label : the env label string (must match filenames exactly)
        """
        paths = sorted(Path(output_dir).glob(f"{label}_seed*.csv"))
        if not paths:
            raise FileNotFoundError(f"No CSVs found for label '{label}' in {output_dir}")
        return pl.concat([pl.read_csv(p) for p in paths])
