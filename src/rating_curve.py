import numpy as np
import pandas as pd


class RatingCurve:
    def __init__(self, flow_cfs, stage_ft):
        self.flow_cfs = np.asarray(flow_cfs, dtype=float)
        self.stage_ft = np.asarray(stage_ft, dtype=float)

    @classmethod
    def from_csv(cls, path):
        df = pd.read_csv(path)
        if not {"flow_cfs", "stage_ft"}.issubset(df.columns):
            raise ValueError("Rating CSV must contain flow_cfs and stage_ft.")
        df = df[["flow_cfs", "stage_ft"]].dropna().sort_values("flow_cfs")
        df = df.drop_duplicates("flow_cfs")
        if len(df) < 2:
            raise ValueError("Rating curve needs at least two points.")
        return cls(df.flow_cfs, df.stage_ft)

    def stage_from_flow(self, flow_cfs):
        q = float(flow_cfs)
        # Explicitly clamp; do not silently extrapolate beyond the published table.
        return float(np.interp(q, self.flow_cfs, self.stage_ft))

    def flow_from_stage(self, stage_ft):
        """Inverse interpolation of the local rating curve, also clamped."""
        h = float(stage_ft)
        order = np.argsort(self.stage_ft)
        stage = self.stage_ft[order]
        flow = self.flow_cfs[order]
        keep = np.concatenate([[True], np.diff(stage) > 1e-9])
        stage = stage[keep]
        flow = flow[keep]
        if len(stage) < 2:
            raise ValueError("Rating curve cannot be inverted because stage values are not unique.")
        return float(np.interp(h, stage, flow))

    @property
    def min_flow_cfs(self):
        return float(np.nanmin(self.flow_cfs))

    @property
    def max_flow_cfs(self):
        return float(np.nanmax(self.flow_cfs))

    @property
    def min_stage_ft(self):
        return float(np.nanmin(self.stage_ft))

    @property
    def max_stage_ft(self):
        return float(np.nanmax(self.stage_ft))
