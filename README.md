# pydaltoolboxdp: Deep ++ Extensions for `pydaltoolbox`

`pydaltoolboxdp` is the Python implementation of the DAL ecosystem's **Deep ++**
layer. It provides deep-learning and advanced models that specialize the contracts
defined by `pydaltoolbox`, `pytspredit`, and `pyharbinger`.

The sibling R package, `daltoolboxdp`, is a thin `reticulate` interface to these
classes. Python algorithms live here once; they are not duplicated inside the R
package.

The project is in its initial porting phase. The implemented vertical slices are the
dense and denoising autoencoder families:

- `AutoencE`: transforms observations into their latent representation;
- `AutoencEd`: reconstructs observations after encoding.
- `AutoencDenoiseE`: learns robust latent representations from corrupted inputs;
- `AutoencDenoiseEd`: reconstructs clean observations after denoising training.

## Installation

Development installation from this checkout:

```bash
python -m pip install -e .
```

On Python 3.14, prefer binary dependencies so `pyharbinger` selects the available
pure-Python `ruptures` wheel instead of attempting a local C compilation:

```bash
python -m pip install --prefer-binary pydaltoolboxdp
```

PyTorch is a core dependency because Deep ++ is specifically the deep-learning
extension layer. GPU use is optional; models run on CPU by default.
