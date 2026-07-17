import numpy as np
import xarray as xr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ensemble_stats import analysis, plotting, synthetic


def test_synthetic_example_dimensions_and_metadata_propagate_to_stl():
    da = synthetic.generate_synthetic_nested_ensemble(
        nt=36,
        nj=2,
        nk=3,
        gseasonal_amp=1.2,
        t_dim="time",
        variable_name="sie_n",
        variable_label="SIE",
        units=r"$10^6$ km$^2$",
        seed=123,
    )

    assert da.dims == ("j", "k", "time")
    assert da.shape == (2, 3, 36)
    assert da.name == "sie_n"
    assert da.attrs["long_name"] == "SIE"
    assert da.attrs["units"] == r"$10^6$ km$^2$"

    ds_stl = analysis.compute_STL_decomposition(da)

    assert ds_stl.attrs["source_variable"] == "sie_n"
    assert ds_stl.attrs["source_long_name"] == "SIE"
    assert ds_stl.attrs["source_units"] == r"$10^6$ km$^2$"
    assert ds_stl.STL_trend.attrs["long_name"] == "STL trend of SIE"
    assert ds_stl.STL_trend.attrs["units"] == r"$10^6$ km$^2$"


def test_stl_plot_metadata_and_residual_overlay_offset():
    time = np.arange(24)
    ds_stl = xr.Dataset(
        {
            "STL_trend": ("time", np.linspace(0.0, 1.0, len(time))),
            "STL_seasonal": ("time", 0.1 * np.sin(time)),
            "STL_resid": ("time", 0.01 * np.cos(time)),
        },
        coords={"time": time},
    )

    metadata = plotting.PlotMetadata(
        variable_label="SIE",
        units=r"$10^6$ km$^2$",
        time_label="Year",
    )

    fig, axes = plotting.plot_STL_decomposition(
        ds_stl,
        metadata=metadata,
        residual_overlay_offset=1.0,
    )

    try:
        assert axes[0, 0].get_xlabel() == "Year"
        assert axes[0, 0].get_ylabel() == r"SIE ($10^6$ km$^2$)"

        _, labels = axes[1, 0].get_legend_handles_labels()
        assert "residual + 1" in labels

        horizontal_lines_at_offset = [
            line
            for line in axes[1, 0].get_lines()
            if np.allclose(line.get_ydata(), 1.0)
        ]
        assert horizontal_lines_at_offset
    finally:
        plt.close(fig)
