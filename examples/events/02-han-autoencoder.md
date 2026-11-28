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

    dense [] {'TP': 0, 'FP': 0, 'FN': 1, 'TN': 100, 'confMatrix':                  event_TRUE  event_FALSE
    detection_TRUE            0            0
    detection_FALSE           1          100, 'accuracy': 0.9900990099009901, 'sensitivity': 0.0, 'specificity': 1.0, 'prevalence': 0.009900990099009901, 'PPV': nan, 'NPV': 0.9900990099009901, 'detection_rate': 0.0, 'detection_prevalence': 0.0, 'balanced_accuracy': 0.5, 'precision': nan, 'recall': 0.0, 'F1': nan}
    denoising [] {'TP': 0, 'FP': 0, 'FN': 1, 'TN': 100, 'confMatrix':                  event_TRUE  event_FALSE
    detection_TRUE            0            0
    detection_FALSE           1          100, 'accuracy': 0.9900990099009901, 'sensitivity': 0.0, 'specificity': 1.0, 'prevalence': 0.009900990099009901, 'PPV': nan, 'NPV': 0.9900990099009901, 'detection_rate': 0.0, 'detection_prevalence': 0.0, 'balanced_accuracy': 0.5, 'precision': nan, 'recall': 0.0, 'F1': nan}
    

    convolutional [] {'TP': 0, 'FP': 0, 'FN': 1, 'TN': 100, 'confMatrix':                  event_TRUE  event_FALSE
    detection_TRUE            0            0
    detection_FALSE           1          100, 'accuracy': 0.9900990099009901, 'sensitivity': 0.0, 'specificity': 1.0, 'prevalence': 0.009900990099009901, 'PPV': nan, 'NPV': 0.9900990099009901, 'detection_rate': 0.0, 'detection_prevalence': 0.0, 'balanced_accuracy': 0.5, 'precision': nan, 'recall': 0.0, 'F1': nan}
    lstm [] {'TP': 0, 'FP': 0, 'FN': 1, 'TN': 100, 'confMatrix':                  event_TRUE  event_FALSE
    detection_TRUE            0            0
    detection_FALSE           1          100, 'accuracy': 0.9900990099009901, 'sensitivity': 0.0, 'specificity': 1.0, 'prevalence': 0.009900990099009901, 'PPV': nan, 'NPV': 0.9900990099009901, 'detection_rate': 0.0, 'detection_prevalence': 0.0, 'balanced_accuracy': 0.5, 'precision': nan, 'recall': 0.0, 'F1': nan}
    

    stacked [] {'TP': 0, 'FP': 0, 'FN': 1, 'TN': 100, 'confMatrix':                  event_TRUE  event_FALSE
    detection_TRUE            0            0
    detection_FALSE           1          100, 'accuracy': 0.9900990099009901, 'sensitivity': 0.0, 'specificity': 1.0, 'prevalence': 0.009900990099009901, 'PPV': nan, 'NPV': 0.9900990099009901, 'detection_rate': 0.0, 'detection_prevalence': 0.0, 'balanced_accuracy': 0.5, 'precision': nan, 'recall': 0.0, 'F1': nan}
    variational [] {'TP': 0, 'FP': 0, 'FN': 1, 'TN': 100, 'confMatrix':                  event_TRUE  event_FALSE
    detection_TRUE            0            0
    detection_FALSE           1          100, 'accuracy': 0.9900990099009901, 'sensitivity': 0.0, 'specificity': 1.0, 'prevalence': 0.009900990099009901, 'PPV': nan, 'NPV': 0.9900990099009901, 'detection_rate': 0.0, 'detection_prevalence': 0.0, 'balanced_accuracy': 0.5, 'precision': nan, 'recall': 0.0, 'F1': nan}
    adversarial [] {'TP': 0, 'FP': 0, 'FN': 1, 'TN': 100, 'confMatrix':                  event_TRUE  event_FALSE
    detection_TRUE            0            0
    detection_FALSE           1          100, 'accuracy': 0.9900990099009901, 'sensitivity': 0.0, 'specificity': 1.0, 'prevalence': 0.009900990099009901, 'PPV': nan, 'NPV': 0.9900990099009901, 'detection_rate': 0.0, 'detection_prevalence': 0.0, 'balanced_accuracy': 0.5, 'precision': nan, 'recall': 0.0, 'F1': nan}
    
