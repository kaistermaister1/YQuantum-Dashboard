# Mock package data generation

The process consists of two steps: first, build vehicles; second, measure take rates from these.

## Vehicle generation

Generate some vehicles:

```bash
python generate_vehicles.py -N 1000 --input_dir data --output_file data/test_vehicles.csv
```

Make sure that the number of cars is large enough, else take rates of pairs or even higher order tuples of options cannot be measured.
The script will randomly generate a take rate for each provided option, and then use that take rate to include (or not include) the option in a vehicle.
Conflicting options are never included.

This will also create test_vehicles_raw_take_rates.csv and test_vehicles_take_rates.csv. (Basically, it relies on the basename of your output filename.)

## Take rate data building

Next, generate some package take rates:

```bash
python build_package_take_rate_data.py --families_file data/families.csv  --options_file data/options.csv  --vehicles_file data/test_vehicles.csv  --template_file package_templates.yaml    --output_file ../pipelines/data/take_rates.parquet --output_format parquet
```

As you can see from the options, it requires some data from `data` to know that options exist, and uses the output `test_vehicles.csv` from the previous step.

Packages that we want to measure take rates for are steered by `package_templates.yaml`, which contains blocks like these:

```yaml
# Lists which families (by fam_abbrev) each package type may draw from.
package_templates:
  InteriorPackage:
    families:
      - FAM_STWHL    # Steering Wheel
      - FAM_STYPE    # Seat Type
      - FAM_SFEAT    # Seat Features
      - FAM_UPHOL    # Upholstery Material
      - FAM_LBA      # Lighting & Ambience (e.g. scuff plates, ambient LEDs)

  PackageTypeTwo:
    families:
      - FAM_ABC
      - FAM_DEF
```

The complexity depends on the number of families you include in an option - in particular, the example will scale as N_seats times N_seat_features times N_upholsteries times N_lightning. (And then again through all possible subsets.) Remember that the number of seats in `FAM_STYPE` is steered by the data in `data/options.csv`:

```csv
opt_abbrev,name,family,cost_eur,price_eur,margin_eur
OPT_STYPE_STD,Standard seats,FAM_STYPE,500,1250,750
OPT_STYPE_SP,Sport seats,FAM_STYPE,650,1600,950
OPT_STYPE_EX,Executive seats,FAM_STYPE,800,2000,1200
```

### Parallel running

If you dataset is too big, experiment with the options `--parallel` for parallel running (which relies on `--num-processes` and `--batch-size`).

If you suffer from memory troubles, try the streaming option `--streaming` to directly stream data to disk.

Note that both options are still experimental and don't fully work as indented quite yet.
