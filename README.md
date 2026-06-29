# MTD — Storm Cell Tracking (Split & Merge)

This code finds rain/storm objects in radar precipitation maps and follows them through
time. It can also handle the cases where one storm **splits** into parts, or where parts
**merge** together. It is based on the MTD object-based tracking method (MODE Time-Domain).

The code runs in **5 steps**. Each step reads the output of the step before it.

---

## How it works (in detail)

The pipeline turns raw precipitation maps into tracked storm objects and a set of properties
for each object and for each storm system. Here is what every component does, and how.

### Step 1 — Convolution: making the object masks
`MTD/obj_convolution.py` works on one precipitation map at a time. It first smooths the rain
field with a **box filter** (a moving average) of size `2R+1 × 2R+1`. Smoothing removes single
noisy pixels and joins nearby rain into solid blobs. It then turns the smoothed field into a
**0/1 mask**: a pixel becomes `1` (inside an object) when its smoothed value is greater than
or equal to `Th`, and `0` otherwise. Importantly, the **original** rain values are kept (as
`fcst_raw`) next to the mask (as `fcst_object_id`) — the smoothing is used only to decide
*where* the objects are, not to change the rain values used later. Maps that contain no object
are skipped, and the time of each map is read from inside the file.

### Step 2 — Merge: joining in time and splitting on gaps
`MTD/obj_merging.py` sorts all the single-time masks by time and groups them by **year** and
**season** (DJF, MAM, JJA, SON); grouping by season keeps each file at a sensible size. The
maps in a group are stacked along the time axis. The code then looks at the **time gap**
between each pair of neighbouring steps: wherever a gap is **larger than**
`time_gap_threshold_hours`, it cuts the series and starts a new piece. The result is that
every output file is one continuous run of time with no large hole — for example, two storms
several hours apart never end up in the same file.

### Step 3 — Tracking: giving each object an ID through time
`MTD/obj_saving_trakced.py` gives every object an **ID that follows it as it moves**, and
records when objects **split** or **merge**. The blobs in the first time step are numbered
with `skimage.label`. For every following step, the **previous** frame and the **current**
frame are stacked and labelled together as a single 3D shape: if a blob in the previous frame
**overlaps** a blob in the current frame, the two form one connected 3D piece and are treated
as the **same object continuing**, so the ID is carried forward. From these overlaps the code
distinguishes four cases — **continue** (one → one, same ID), **split** (one → many, new IDs
for the new pieces), **merge** (many → one, a new ID), and **new** (a blob with no overlap, a
fresh ID). Every split and merge is stored as a parent→child pair of IDs. In the output the
object field (`fcst_object_id`) now holds these track IDs.

### Step 4 — Object properties: measuring each object
`MTD/obj_object_analysis.py` measures every object at every time with `skimage.regionprops`:
**area**, **centroid** (position), **orientation**, and **aspect ratio** (minor axis ÷ major
axis). From the raw rain inside each object it computes `Ismax` (the strongest rain) and `Iv`
(the total rain, i.e. the sum over the object's pixels), and it flags objects that **touch the
domain edge**. Objects smaller than `area_threshold`, or with no rain inside, are removed. The
code then follows each ID from one time to the next to get its **movement** — the distance
moved (`d`), the speed (`Velocity`) and the direction (`dir`) — using the centroids together
with `pixel_resolution` and `time_resolution`; area, intensity and shape are averaged over the
two times. The single-time measurements form the **snapshot** table; the between-time movement
forms the **averaged** table.

### Step 5 — System properties: building the full storm tracks
`MTD/obj_system_analysis.py` joins the objects into complete storms. Because a storm can split
and merge many times, its whole life is a **network**, not a single line. The code builds a
**directed graph** in which each node is an object ID and each edge is a split/merge link, with
the edge weighted by the distance that object travelled (`d`). For each separate system (each
connected group in the graph) it adds a helper end node and finds the **longest path** through
the network; the total distance along that path is the **storm track length** — how far the
longest branch of the storm travelled in all.

---

## Folder structure

```
MTD-cell_tracking_split_merge/
├── MTD/                       # the code (the 5 steps)
├── MRMS-Sample_data/
│   ├── inputs/                # input files (you provide these)
│   └── outputs/               # all results (created automatically)
├── run/
│   ├── example.py             # runs all 5 steps
│   └── example.pbs            # submits the job on NCI (PBS)
└── README.md
```

---

## Input files — what they must look like

Please make sure your input files follow these rules, or the code will not run:

- **Format:** NetCDF (`.nc`), readable by `xarray`.
- **One time step per file.** Each file must hold only **one** time. The file must still
  have a `time` dimension/coordinate (with length 1). If a file has more than one time step,
  the code will fail. Splitting your data into one-step files is a preprocessing step you do
  yourself.
- **Variable name:** each file must contain the precipitation variable named
  **`PrecipRate_0mabovemeansealevel`** — the rain rate on a 2D grid (latitude × longitude).
  If your variable has another name, change `input_raster_main_field` in `run/example.py`
  (Step 1).
- **Time coordinate name:** the time coordinate must be named **`time`**. If yours is
  different, change `input_raster_time_field` in `run/example.py` (Step 1).
- **Same grid:** all input files must use the **same** spatial grid (same shape), so they
  can be joined together in time.
- **Folder content:** put **only** the input `.nc` files in `MRMS-Sample_data/inputs/`. The
  code reads *every* file in that folder, so do not put other files there.
- **File name:** the file name is **not** important. The code reads the time from *inside*
  the file, not from the name. (The sample files are named like
  `PrecipRate_00.00_20141101-153000_double_double.nc`.)

---

## Outputs — what you get

All outputs are written under `MRMS-Sample_data/outputs/`. The folders are created
automatically.

| Folder / file | Step | Format | What is inside |
|---|---|---|---|
| `convolved_maps/convolved_YYYYMMDDTHHMM.nc` | 1 | NetCDF, one per time step | `fcst_raw` (original rain rate) and `fcst_object_id` (object mask) |
| `merged_convolved_maps/{first}_{last}.nc` | 2 | NetCDF | Convolved maps joined in time, split by season and by time gaps |
| `merged_convolved_maps_tracked/{first}_{last}.nc` | 3 | NetCDF | Same as merged, but `fcst_object_id` now holds the **track ID** of each object |
| `connections_folder/connections.npy` | 3 | NumPy | Which objects split or merge |
| `objects/snapshot_properties/Radar_data_obj_{0-3}.ftr` | 4 | Feather (pandas) | Object properties at each single time |
| `objects/averaged_properties/Radar_data_ave_{0-3}.ftr` | 4 | Feather (pandas) | Object movement properties (between two times) |
| `systems/stormtracklength_list.pkl` | 5 | Pickle | Storm track lengths |

The number `{0-3}` is the season: **0 = SON, 1 = DJF, 2 = MAM, 3 = JJA**.

### Property columns (Step 4 tables)

| Column | Meaning |
|---|---|
| `label` | object / track ID |
| `datetime` | time of the snapshot |
| `Centroid_X`, `Centroid_Y` | object centre position (on the grid) |
| `area` | object size |
| `Ismax` | maximum rain rate inside the object |
| `Iave` | average rain rate inside the object |
| `Iv` | "intensity volume" = sum of the rain rate over the object |
| `aspectratio` | shape: minor axis / major axis |
| `orientation` | object orientation angle (degrees) |
| `d` | distance the object moved between two times |
| `Velocity` | object speed |
| `dir` | object direction |
| `touched_borders?` | `True` if the object touches the edge of the domain |

(`d`, `Velocity` and `dir` are in the **averaged** table only.)

---

## Parameters you can change

You set these in `run/example.py`:

| Parameter | Step | Meaning | Sample value |
|---|---|---|---|
| `R` | 1 | smoothing radius (the filter window is `2R+1`) | `3` |
| `Th` | 1 | threshold to make the object mask | `0.1` |
| `time_gap_threshold_hours` | 2 | largest time gap allowed inside one merged file (hours) | `2` |
| `area_threshold` | 4 | smallest object size to keep (pixels) | `1` |
| `pixel_resolution` | 4 | grid spacing in km | `10` |
| `time_resolution` | 4 | time step in minutes | `30` |

> If your data has a different time step (for example 5 minutes), choose a suitable
> `time_gap_threshold_hours` for Step 2 and set the correct `time_resolution` for Step 4.

---

## How to run

### On NCI (PBS job)

```bash
cd run
qsub example.pbs
```

`example.pbs` loads the Python environment and runs `example.py` (all 5 steps). The
environment is loaded with:

```bash
module use /g/data/xp65/public/modules
module load conda/analysis3
```

Change `-P` (project), `storage`, `mem` and `walltime` in `example.pbs` to match your
project and the size of your data.

### As a plain script

With a suitable Python environment active:

```bash
cd run
python3 example.py
```

Outputs go to `MRMS-Sample_data/outputs/`.

---

## Requirements

Python with: `xarray`, `numpy`, `pandas`, `scipy`, `scikit-image`, `netCDF4`, `networkx`,
`pyarrow`. On NCI, the `analysis3` environment (loaded above) already has all of these.

---

## References

- https://link.springer.com/article/10.1007/s00382-022-06404-z
- https://journals.ametsoc.org/view/journals/hydr/22/1/jhm-d-20-0187.1.xml
- https://www.science.org/doi/abs/10.1126/science.abn8657

## Author

Hooman Ayat
