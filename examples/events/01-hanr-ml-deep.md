# Deep forecasting residuals for anomaly detection

`HanrMl` turns one-step forecast errors into anomaly scores. Here the generic
`pyharbinger` orchestration receives the Deep ++ LSTM and Conv1D forecasters.


```python
from pyharbinger import HanrMl, datasets
from pytspredit import TsNormGminmax
from pydaltoolboxdp import TsConv1D, TsLstm

dataset = datasets.examples_anomalies()["simple"]
serie = dataset["serie"]
event = dataset["event"]
window = 7

models = {
    "LSTM": TsLstm(
        TsNormGminmax(), input_size=window - 1, hidden_size=4,
        sequence_length=window - 1, epochs=3, batch_size=8, random_state=401,
    ),
    "Conv1D": TsConv1D(
        TsNormGminmax(), input_size=window - 1, conv_channels=(4,),
        kernel_sizes=(2,), strides=(1,), dense_hidden_sizes=(4,),
        epochs=3, batch_size=8, random_state=403,
    ),
}

for name, forecaster in models.items():
    detector = HanrMl(forecaster, sw_size=window).fit(serie)
    detection = detector.detect(serie)
    metrics = detector.evaluate(detection, event)
    print(name, detection.index[detection["event"]].tolist(), metrics)
```

    LSTM [] {'TP': 0, 'FP': 0, 'FN': 1, 'TN': 100, 'confMatrix':                  event_TRUE  event_FALSE
    detection_TRUE            0            0
    detection_FALSE           1          100, 'accuracy': 0.9900990099009901, 'sensitivity': 0.0, 'specificity': 1.0, 'prevalence': 0.009900990099009901, 'PPV': nan, 'NPV': 0.9900990099009901, 'detection_rate': 0.0, 'detection_prevalence': 0.0, 'balanced_accuracy': 0.5, 'precision': nan, 'recall': 0.0, 'F1': nan}
    Conv1D [] {'TP': 0, 'FP': 0, 'FN': 1, 'TN': 100, 'confMatrix':                  event_TRUE  event_FALSE
    detection_TRUE            0            0
    detection_FALSE           1          100, 'accuracy': 0.9900990099009901, 'sensitivity': 0.0, 'specificity': 1.0, 'prevalence': 0.009900990099009901, 'PPV': nan, 'NPV': 0.9900990099009901, 'detection_rate': 0.0, 'detection_prevalence': 0.0, 'balanced_accuracy': 0.5, 'precision': nan, 'recall': 0.0, 'F1': nan}
    
