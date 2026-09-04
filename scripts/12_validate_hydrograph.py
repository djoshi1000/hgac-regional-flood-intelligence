from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.validation import validate_hydrograph

p = argparse.ArgumentParser(description="Compare historical NWM/reanalysis hydrograph with observed USGS/HCFCD discharge.")
p.add_argument("--event-name", required=True)
p.add_argument("--modeled-csv", required=True)
p.add_argument("--observed-csv", required=True)
p.add_argument("--output-dir", default="outputs/validation")
p.add_argument("--modeled-time-col", default="time")
p.add_argument("--modeled-flow-col", default="flow_cfs")
p.add_argument("--observed-time-col", default="time")
p.add_argument("--observed-flow-col", default="flow_cfs")
p.add_argument("--modeled-unit", default="cfs")
p.add_argument("--observed-unit", default="cfs")
p.add_argument("--tolerance-minutes", type=float, default=30.0)
args = p.parse_args()

out = Path(args.output_dir) / args.event_name
result = validate_hydrograph(
    modeled_csv=args.modeled_csv,
    observed_csv=args.observed_csv,
    output_dir=out,
    event_name=args.event_name,
    modeled_time_col=args.modeled_time_col,
    modeled_flow_col=args.modeled_flow_col,
    observed_time_col=args.observed_time_col,
    observed_flow_col=args.observed_flow_col,
    modeled_unit=args.modeled_unit,
    observed_unit=args.observed_unit,
    tolerance_minutes=args.tolerance_minutes,
)
print("\n========================================")
print("V4.3 HISTORICAL HYDROGRAPH VALIDATION")
print("========================================")
print(f"Matched points:             {result['matched_points']}")
print(f"RMSE:                       {result['rmse_cfs']:.1f} cfs")
print(f"MAE:                        {result['mae_cfs']:.1f} cfs")
print(f"Bias model-observed:        {result['bias_cfs_model_minus_observed']:.1f} cfs")
print(f"Percent bias:               {result['percent_bias'] if result['percent_bias'] is not None else 'N/A'}")
print(f"Pearson r:                  {result['pearson_r'] if result['pearson_r'] is not None else 'N/A'}")
print(f"NSE:                        {result['nash_sutcliffe_efficiency'] if result['nash_sutcliffe_efficiency'] is not None else 'N/A'}")
print(f"KGE:                        {result['kling_gupta_efficiency'] if result['kling_gupta_efficiency'] is not None else 'N/A'}")
print(f"Observed/modeled peak:      {result['observed_peak_cfs']:.1f} / {result['modeled_peak_cfs']:.1f} cfs")
print(f"Peak timing error:          {result['peak_timing_error_hours_model_minus_observed']:.2f} h (model-observed)")
