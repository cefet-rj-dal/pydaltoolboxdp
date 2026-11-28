---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Autoencoder anomaly detectors

`HanAutoencoder` learns normal sliding windows and flags unusually large
reconstruction errors. The orchestration is shared while the representation
changes across the seven Deep ++ encoder-decoder families.

```python
from pyharbinger import datasets
from pydaltoolboxdp import (
    AutoencAdvEd,
    AutoencConvEd,
    AutoencDenoiseEd,
    AutoencEd,
    AutoencLstmEd,
    AutoencStackedEd,
    AutoencVariationalEd,
    HanAutoencoder,
)

dataset = datasets.examples_anomalies()["simple"]
serie = dataset["serie"]
event = dataset["event"]

families = {
    "dense": (AutoencEd, {}),
    "denoising": (AutoencDenoiseEd, {"noise_factor": 0.1}),
    "convolutional": (
        AutoencConvEd,
        {"hidden_channels": 4, "bottleneck_channels": 4},
    ),
    "lstm": (AutoencLstmEd, {"sequence_length": 6, "lstm_hidden_size": 4}),
    "stacked": (
        AutoencStackedEd,
        {"encoding_sizes": (4, 3), "encoder_hidden_sizes": ((8,), (6,))},
    ),
    "variational": (AutoencVariationalEd, {}),
    "adversarial": (AutoencAdvEd, {}),
}

for offset, (name, (encoder_class, options)) in enumerate(families.items()):
    detector = HanAutoencoder(
        6,
        3,
        encoder_class=encoder_class,
        epochs=3,
        batch_size=8,
        random_state=410 + offset,
        **options,
    ).fit(serie)
    detection = detector.detect(serie)
    metrics = detector.evaluate(detection, event)
    print(name, detection.index[detection["event"]].tolist(), metrics)
```
