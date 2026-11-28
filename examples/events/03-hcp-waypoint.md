# Adaptive Waypoint change-point detection

Waypoint supervises autoencoder reconstruction error with bilateral CUSUM and
relearns the representation after a confirmed regime change. We compare dense,
LSTM, and Conv1D encoders under the same detector settings.


```python
from pyharbinger import datasets
from pydaltoolboxdp import AutoencConvEd, AutoencEd, AutoencLstmEd, HcpWaypoint

dataset = datasets.examples_changepoints()["simple"]
serie = dataset["serie"]
event = dataset["event"]

families = {
    "dense": (AutoencEd, {}),
    "lstm": (AutoencLstmEd, {"sequence_length": 6, "lstm_hidden_size": 4}),
    "convolutional": (
        AutoencConvEd,
        {"hidden_channels": 4, "bottleneck_channels": 4},
    ),
}

for offset, (name, (encoder_class, options)) in enumerate(families.items()):
    detector = HcpWaypoint(
        6,
        3,
        warmup=40,
        retrain_size=20,
        buffer_size=25,
        h_low=1.5,
        h_high=3.0,
        epochs_init=3,
        epochs_retrain=2,
        encoder_class=encoder_class,
        epochs=3,
        batch_size=8,
        random_state=430 + offset,
        **options,
    ).fit(serie)
    detection = detector.detect(serie)
    metrics = detector.evaluate(detection, event)
    print(name, detector.drift_indices_, detector.tau_, metrics)
```

    dense [93] 0.6358322683844724 {'TP': 0, 'FP': 1, 'FN': 1, 'TN': 99, 'confMatrix':                  event_TRUE  event_FALSE
    detection_TRUE            0            1
    detection_FALSE           1           99, 'accuracy': 0.9801980198019802, 'sensitivity': 0.0, 'specificity': 0.99, 'prevalence': 0.009900990099009901, 'PPV': 0.0, 'NPV': 0.99, 'detection_rate': 0.0, 'detection_prevalence': 0.009900990099009901, 'balanced_accuracy': 0.495, 'precision': 0.0, 'recall': 0.0, 'F1': nan}
    lstm [94] 0.3797072955062764 {'TP': 0, 'FP': 1, 'FN': 1, 'TN': 99, 'confMatrix':                  event_TRUE  event_FALSE
    detection_TRUE            0            1
    detection_FALSE           1           99, 'accuracy': 0.9801980198019802, 'sensitivity': 0.0, 'specificity': 0.99, 'prevalence': 0.009900990099009901, 'PPV': 0.0, 'NPV': 0.99, 'detection_rate': 0.0, 'detection_prevalence': 0.009900990099009901, 'balanced_accuracy': 0.495, 'precision': 0.0, 'recall': 0.0, 'F1': nan}
    

    convolutional [97] 0.2215536843224822 {'TP': 0, 'FP': 1, 'FN': 1, 'TN': 99, 'confMatrix':                  event_TRUE  event_FALSE
    detection_TRUE            0            1
    detection_FALSE           1           99, 'accuracy': 0.9801980198019802, 'sensitivity': 0.0, 'specificity': 0.99, 'prevalence': 0.009900990099009901, 'PPV': 0.0, 'NPV': 0.99, 'detection_rate': 0.0, 'detection_prevalence': 0.009900990099009901, 'balanced_accuracy': 0.495, 'precision': 0.0, 'recall': 0.0, 'F1': nan}
    
