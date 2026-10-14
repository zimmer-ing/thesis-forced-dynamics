# Tests

Run the small CPU checks from the repository root in the installed environment:

```sh
python -m unittest discover -s tests -v
```

No downloaded datasets, checkpoints or GPU are needed. CSV fixtures are created
in a temporary directory and removed after the test.

| File | Check |
| --- | --- |
| `test_model_interface.py` | Forecast shape, finite values and backward gradients for FC and GRU with synthetic inputs. |
| `test_dataset_interface.py` | Channel mapping, history/forecast windows, file and split boundaries, unchanged prepared values and batch shapes. |
| `test_fixed_step_fdeint_enforced.py` | A short PECE reference case for the fractional solver, using alpha = 1 and no external forcing. |

## Adding a model

Register the model in `src/models/__init__.py` and add its YAML configuration as
described in [Code structure](../docs/Code_structure.md). Add its name to
`MODEL_NAMES` in `test_model_interface.py` to run the same interface check.
The test uses the configuration defaults and a short synthetic dataset with two
input channels and three output channels. Adjust these dimensions and window
lengths if your model has specific requirements. Physics-based models may also
need their physics configuration. The backward check is for trainable models.

## Adding a dataset

`ExampleDataset` in `test_dataset_interface.py` is a small CSV example derived
from `MultiFileTimeSeriesDataset`. Copy the test for a new dataset and adapt its
file layout, constructor, column names and known expected values to your loader.
Keep the fixtures small and check the train, validation and test split
separately. The example uses values prepared before loading. It does not fit a
scaler.

These checks help catch interface errors. They do not check prediction accuracy,
the full training workflow or reproduction of dissertation results.
