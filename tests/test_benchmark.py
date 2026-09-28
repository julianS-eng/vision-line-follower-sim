import csv
from pathlib import Path

from vision_line_follower.benchmark.run_benchmark import (
    BenchmarkCase,
    BenchmarkConfig,
    BenchmarkRecord,
    NoiseLevel,
    aggregate_results,
    build_default_cases,
    run_benchmark,
    write_aggregate_csv,
    write_records_csv,
)


def _fake_record(
    controller: str, noise: str, track: str, success: bool, err: float
) -> BenchmarkRecord:
    case = BenchmarkCase(
        track_name=track, noise_level=noise, controller_name=controller, run_seed=0
    )
    return BenchmarkRecord(
        track_name=case.track_name,
        noise_level=case.noise_level,
        controller_name=case.controller_name,
        run_seed=case.run_seed,
        success=success,
        failure_reason=None if success else "off_track",
        laps_completed=1.0 if success else 0.4,
        sim_time_s=10.0,
        mean_abs_cross_track_error_m=err,
        max_abs_cross_track_error_m=err * 2,
        rms_cross_track_error_m=err * 1.2,
        mean_speed_mps=0.3,
        vision_found_rate=0.99,
    )


def test_build_default_cases_count() -> None:
    config = BenchmarkConfig(
        track_names=("oval", "figure_eight"),
        controller_names=("pid", "stanley"),
        noise_levels=(NoiseLevel("clean", 0.1, 0.0, 4.0, 2.0, 0.0),),
        seeds_per_case=2,
    )
    cases = build_default_cases(config)
    assert len(cases) == 2 * 2 * 1 * 2


def test_aggregate_results_groups_correctly() -> None:
    records = [
        _fake_record("pid", "clean", "oval", True, 0.01),
        _fake_record("pid", "clean", "figure_eight", True, 0.02),
        _fake_record("pid", "harsh", "oval", False, 0.05),
    ]
    rows = aggregate_results(records, group_by=("controller_name", "noise_level"))
    by_key = {(r.controller_name, r.noise_level): r for r in rows}
    assert by_key[("pid", "clean")].n_runs == 2
    assert by_key[("pid", "clean")].success_rate == 1.0
    assert by_key[("pid", "harsh")].success_rate == 0.0


def test_write_records_csv_roundtrip(tmp_path: Path) -> None:
    records = [_fake_record("pid", "clean", "oval", True, 0.01)]
    out = tmp_path / "records.csv"
    write_records_csv(records, out)
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["controller_name"] == "pid"


def test_write_aggregate_csv(tmp_path: Path) -> None:
    records = [_fake_record("stanley", "moderate", "curvy_loop", True, 0.015)]
    rows = aggregate_results(records)
    out = tmp_path / "aggregate.csv"
    write_aggregate_csv(rows, out)
    with out.open() as f:
        read_rows = list(csv.DictReader(f))
    assert len(read_rows) == 1


def test_run_benchmark_end_to_end_smoke() -> None:
    config = BenchmarkConfig(
        track_names=("oval",),
        controller_names=("pid",),
        noise_levels=(NoiseLevel("clean", 0.1, 0.0, 4.0, 1.0, 0.0),),
        seeds_per_case=1,
        max_steps=80,
    )
    records = run_benchmark(config)
    assert len(records) == 1
    assert records[0].controller_name == "pid"
    assert records[0].track_name == "oval"
