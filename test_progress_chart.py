"""
Weekly test progress chart for MoE Training tests on XPU.
Generates a line chart showing passing test rate (%) per test file over time.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Total number of tests per file (excluding skips)
test_totals = {
    "ep/test_a2a_dispatch.py": 1,
    "ep/test_compile.py": 1,
    "ep/test_integration.py": 1,
    "ep/test_kernels.py": 9,
    "ep/test_permute.py": 1,
    "mxfp8/test_mxfp8_a2a.py": 2,
    "test_distributed.py": 30,
    "test_fp8_grouped_mm.py": 12,
    "test_kernels.py": 492,
    "test_mxfp8_grouped_mm.py": 145,
    "test_nvfp4_grouped_mm.py": 14,
    "test_tensor.py": 37,
    "test_training.py": 48,
}

# Weekly data points: (date, {test_file: pass_count})
weekly_data = [
    (
        "2026-04-08",
        {
            "ep/test_a2a_dispatch.py": 1,
            "ep/test_compile.py": 0,
            "ep/test_integration.py": 0,
            "ep/test_kernels.py": 9,
            "ep/test_permute.py": 1,
            "mxfp8/test_mxfp8_a2a.py": 0,
            "test_distributed.py": 1,
            "test_fp8_grouped_mm.py": 8,
            "test_kernels.py": 116,
            "test_mxfp8_grouped_mm.py": 16,
            "test_nvfp4_grouped_mm.py": 14,
            "test_tensor.py": 15,
            "test_training.py": 6,
        },
    ),
    (
        "2026-04-22",
        {
            "ep/test_a2a_dispatch.py": 1,
            "ep/test_compile.py": 0,
            "ep/test_integration.py": 0,
            "ep/test_kernels.py": 9,
            "ep/test_permute.py": 1,
            "mxfp8/test_mxfp8_a2a.py": 0,
            "test_distributed.py": 1,
            "test_fp8_grouped_mm.py": 8,
            "test_kernels.py": 118,
            "test_mxfp8_grouped_mm.py": 17,
            "test_nvfp4_grouped_mm.py": 14,
            "test_tensor.py": 15,
            "test_training.py": 6,
        },
    ),
]

# Parse dates
dates = [datetime.strptime(d[0], "%Y-%m-%d") for d in weekly_data]
test_files = list(weekly_data[0][1].keys())

# Compute pass rates (%)
def pass_rate(pass_count, test_file):
    total = test_totals[test_file]
    return (pass_count / total * 100) if total > 0 else 0.0

fig, ax = plt.subplots(figsize=(14, 8))
fig.suptitle("MoE Training XPU Tests — Weekly Pass Rate (%)", fontsize=14, fontweight="bold")

colors = plt.cm.tab20.colors

for idx, test in enumerate(test_files):
    values = [pass_rate(w[1][test], test) for w in weekly_data]
    ax.plot(dates, values, marker="o", label=test, color=colors[idx % len(colors)], linewidth=2)

    # Annotate the latest point with +/- diff if it changed from the previous week
    if len(values) >= 2:
        diff = values[-1] - values[-2]
        if abs(diff) > 0.01:
            sign = "+" if diff > 0 else ""
            ax.annotate(
                f"{sign}{diff:.1f}%",
                xy=(dates[-1], values[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=7,
                fontweight="bold",
                color="green" if diff > 0 else "red",
                va="center",
            )

ax.set_ylabel("Pass Rate (%)")
ax.set_ylim(-5, 105)
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.set_xlabel("Date")

plt.tight_layout()
plt.savefig("test_progress_chart.png", dpi=150, bbox_inches="tight")
print("Chart saved to test_progress_chart.png")
plt.show()
