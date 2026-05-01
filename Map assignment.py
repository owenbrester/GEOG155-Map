"""
GEOG 155 - Stage 2: Map Creation
Dot Density Map: Internet Access by US County
Data Source: US Census Bureau American Community Survey (ACS) 5-Year Estimates
             Table B28002: Presence and Types of Internet Subscriptions in Household
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import geopandas as gpd
import pandas as pd
import requests
import os

SHAPEFILE_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2022/shp/"
    "cb_2022_us_county_5m.zip"
)
SHAPEFILE_PATH = "cb_2022_us_county_5m.zip"

if not os.path.exists(SHAPEFILE_PATH):
    print("Downloading county shapefile...")
    r = requests.get(SHAPEFILE_URL, timeout=60)
    with open(SHAPEFILE_PATH, "wb") as f:
        f.write(r.content)
    print("Download complete.")

counties = gpd.read_file(f"zip://{SHAPEFILE_PATH}")

print("Fetching ACS internet access data...")

ACS_URL = (
    "https://api.census.gov/data/2022/acs/acs5"
    "?get=B28002_001E,B28002_004E"
    "&for=county:*"
    "&in=state:*"
)

response = requests.get(ACS_URL, timeout=60)
acs_data = response.json()

acs_df = pd.DataFrame(acs_data[1:], columns=acs_data[0])
acs_df = acs_df.rename(columns={
    "B28002_001E": "total_households",
    "B28002_004E": "broadband_households",
})
acs_df["total_households"]     = pd.to_numeric(acs_df["total_households"],     errors="coerce")
acs_df["broadband_households"] = pd.to_numeric(acs_df["broadband_households"], errors="coerce")
acs_df["GEOID"] = acs_df["state"] + acs_df["county"]

print(f"ACS data loaded: {len(acs_df)} counties.")

counties = counties.merge(acs_df[["GEOID", "total_households", "broadband_households"]],
                          on="GEOID", how="left")

EXCLUDED = ["72", "78", "60", "66", "69"]
counties = counties[~counties["STATEFP"].isin(EXCLUDED)].copy()
counties = counties[counties["total_households"] > 0].copy()

counties["no_broadband"] = counties["total_households"] - counties["broadband_households"]
counties["no_broadband"]  = counties["no_broadband"].clip(lower=0)

counties = counties.to_crs("ESRI:102003")

DOT_VALUE = 500
np.random.seed(42)

def random_points_in_polygon(polygon, n):
    if n <= 0:
        return np.empty((0, 2))
    minx, miny, maxx, maxy = polygon.bounds
    pts = []
    attempts = 0
    while len(pts) < n and attempts < n * 20:
        xs = np.random.uniform(minx, maxx, (n - len(pts)) * 4)
        ys = np.random.uniform(miny, maxy, (n - len(pts)) * 4)
        for x, y in zip(xs, ys):
            from shapely.geometry import Point
            if polygon.contains(Point(x, y)):
                pts.append((x, y))
            if len(pts) == n:
                break
        attempts += n * 4
    return np.array(pts) if pts else np.empty((0, 2))

print("Placing dots (this may take ~1–2 minutes)...")

broadband_x, broadband_y = [], []
no_broadband_x, no_broadband_y = [], []

for _, row in counties.iterrows():
    geom = row.geometry
    if geom is None or geom.is_empty:
        continue

    n_broad  = max(0, int(round(row["broadband_households"] / DOT_VALUE)))
    n_no     = max(0, int(round(row["no_broadband"]         / DOT_VALUE)))

    if n_broad > 0:
        pts = random_points_in_polygon(geom, n_broad)
        if len(pts):
            broadband_x.extend(pts[:, 0])
            broadband_y.extend(pts[:, 1])

    if n_no > 0:
        pts = random_points_in_polygon(geom, n_no)
        if len(pts):
            no_broadband_x.extend(pts[:, 0])
            no_broadband_y.extend(pts[:, 1])

print(f"Dots placed — broadband: {len(broadband_x):,} | no broadband: {len(no_broadband_x):,}")

fig, ax = plt.subplots(figsize=(18, 11))
fig.patch.set_facecolor("#0d1b2a")
ax.set_facecolor("#0d1b2a")

counties.plot(ax=ax, color="none", edgecolor="#2a3f54", linewidth=0.15, zorder=1)

states = counties.dissolve(by="STATEFP").reset_index()
states.plot(ax=ax, color="none", edgecolor="#7aaabb", linewidth=0.6, zorder=4)

STATE_NAMES = {
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT",
    "10":"DE","11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL",
    "18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD",
    "25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE",
    "32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND",
    "39":"OH","40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD",
    "47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV",
    "55":"WI","56":"WY",
}
for _, row in states.iterrows():
    abbr = STATE_NAMES.get(row["STATEFP"])
    if abbr and row.geometry and not row.geometry.is_empty:
        cx = row.geometry.centroid.x
        cy = row.geometry.centroid.y
        ax.text(cx, cy, abbr, fontsize=5.5, color="white", alpha=0.75,
                ha="center", va="center", fontweight="bold", zorder=5,
                fontfamily="DejaVu Sans")

DOT_SIZE = 0.3
ax.scatter(no_broadband_x,  no_broadband_y,  s=DOT_SIZE, c="#e05c5c",
           alpha=0.7, linewidths=0, zorder=2, rasterized=True)
ax.scatter(broadband_x,     broadband_y,     s=DOT_SIZE, c="#4fc3f7",
           alpha=0.6, linewidths=0, zorder=3, rasterized=True)

ax.set_axis_off()

fig.text(0.5, 0.95, "Broadband Internet Access Across U.S. Counties",
         ha="center", va="top", fontsize=20, fontweight="bold",
         color="white", fontfamily="DejaVu Sans")

fig.text(0.5, 0.915, "Each dot represents 500 households  ·  ACS 5-Year Estimates, 2022",
         ha="center", va="top", fontsize=10, color="#aaaaaa", fontfamily="DejaVu Sans")

legend_handles = [
    mlines.Line2D([], [], color="#4fc3f7", marker="o", linestyle="None",
                  markersize=8, label="With Broadband"),
    mlines.Line2D([], [], color="#e05c5c", marker="o", linestyle="None",
                  markersize=8, label="Without Broadband"),
]
legend = ax.legend(
    handles=legend_handles,
    loc="lower left",
    frameon=True,
    framealpha=0.25,
    facecolor="#0d1b2a",
    edgecolor="#4fc3f7",
    fontsize=11,
    labelcolor="white",
    title="Legend",
    title_fontsize=11,
)
legend.get_title().set_color("white")

ax.annotate("N", xy=(0.97, 0.175), xycoords="axes fraction",
            ha="center", va="bottom", fontsize=11, color="white", fontweight="bold")
ax.annotate("S", xy=(0.97, 0.065), xycoords="axes fraction",
            ha="center", va="top", fontsize=11, color="white", fontweight="bold")
ax.annotate("W", xy=(0.945, 0.12), xycoords="axes fraction",
            ha="right", va="center", fontsize=11, color="white", fontweight="bold")
ax.annotate("E", xy=(0.995, 0.12), xycoords="axes fraction",
            ha="left", va="center", fontsize=11, color="white", fontweight="bold")
ax.annotate("▲", xy=(0.97, 0.155), xycoords="axes fraction",
            ha="center", va="top", fontsize=13, color="white")

scale_x_start = 0.68
scale_x_end   = 0.85
scale_y       = 0.045
ax.annotate("", xy=(scale_x_end, scale_y), xycoords="axes fraction",
            xytext=(scale_x_start, scale_y),
            arrowprops=dict(arrowstyle="-", color="white", lw=1.5))
for tick_x in [scale_x_start, (scale_x_start + scale_x_end) / 2, scale_x_end]:
    ax.annotate("|", xy=(tick_x, scale_y), xycoords="axes fraction",
                ha="center", va="center", color="white", fontsize=8)
ax.text(scale_x_start + (scale_x_end - scale_x_start) / 2, scale_y - 0.025,
        "~500 km", transform=ax.transAxes,
        ha="center", va="top", fontsize=9, color="white")

fig.text(0.5, 0.015,
         "Data: U.S. Census Bureau, American Community Survey 5-Year Estimates (2022), Table B28002  |  "
         "Geography: TIGER/Line Shapefiles (2022)  |  Author: Owen Brester",
         ha="center", va="bottom", fontsize=7.5, color="#888888")

plt.tight_layout(rect=[0, 0.03, 1, 0.93])

_pan_start = {}

def on_scroll(event):
    if event.inaxes != ax:
        return
    scale = 0.85 if event.button == "up" else 1.0 / 0.85
    cx, cy = event.xdata, event.ydata
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim([cx + (x - cx) * scale for x in xlim])
    ax.set_ylim([cy + (y - cy) * scale for y in ylim])
    fig.canvas.draw_idle()

def on_press(event):
    if event.inaxes != ax or event.button != 1:
        return
    _pan_start["x"] = event.xdata
    _pan_start["y"] = event.ydata

def on_motion(event):
    """Pan the map as the mouse is dragged."""
    if event.inaxes != ax or event.button != 1:
        return
    if "x" not in _pan_start:
        return
    dx = _pan_start["x"] - event.xdata
    dy = _pan_start["y"] - event.ydata
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
    ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
    fig.canvas.draw_idle()

def on_release(event):
    _pan_start.clear()

fig.canvas.mpl_connect("scroll_event",          on_scroll)
fig.canvas.mpl_connect("button_press_event",    on_press)
fig.canvas.mpl_connect("motion_notify_event",   on_motion)
fig.canvas.mpl_connect("button_release_event",  on_release)

print("\nControls: Scroll wheel = zoom  |  Click & drag = pan")

OUTPUT_FILE = "internet_access_dot_density.pdf"
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Map saved to: {OUTPUT_FILE}")
plt.show()
