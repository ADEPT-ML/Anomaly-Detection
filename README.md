# Anomaly-Detection 🩺

The service is responsible for detecting anomalies in the (multivariate) time-series data.
It provides different algorithms for the detection.

## Requirements

+ Python ≥ 3.10
+ All packages from requirements.txt

## Available models

* Isolation Forest
* One-Class-SVM
* DAGMM
* LSTM-Autoencoder

## Development

### Local

Install dependencies from requirements.txt

Start the service:

```sh
uvicorn main:app --reload
```

### Docker

We provide a docker-compose in the root directory of ADEPT to start all services bundled together.

## Adding functionality

New algorithms can be easily added by following the instructions below.
Additional features such as configuration options and on-demand training are also explained.

### Directory structure

```
\-Anomaly-Detection
    ├── algorithms                              # Contains all files for the anomaly detection
    │   ├── interface                           # Contains all interfaces
    │   │   ├── algorithm_config.py
    │   │   ├── algorithm_information.py
    │   │   └── algorithm_interface.py
    │   ├── models                              # Contains trained models and training code
    │   │   ├── DAGMM                           # Contains all files for the DAGMM model
    │   │   │   ├── train                       # Contains the training code for DAGMM
    │   │   │   └── universal                   # Contains a generic pre-trained DAGMM model
    │   │   └── [...]
    │   ├── util                                # Contains useful functions for anomaly detection
    │   │   ├── window_util.py
    │   │   └── [...]
    │   ├── dagmm.py                            # DAGMM Implementation
    │   ├── ocsvm.py                            # OCSVM Implementation
    │   └── [...]
    ├── src                                     # Python source files for base functions
    │   ├── dynamic_algorithm_loading.py
    │   └── [...]
    ├── Dockerfile
    ├── main.py                                 # Main module with all API definitions
    ├── requirements.txt                        # Required python dependencies
    └── [...]
```

### Adding anomaly detection algorithms

1. Create a python module in the [algorithms package](algorithms)
2. Create a class named `Algorithm` that implements
   the [algorithm interface](algorithms/interface/algorithm_interface.py)
3. Implement the anomaly detection algorithms and expose its functions via the interface

### Configuring the anomaly detection algorithm

#### Metadata

The algorithm metadata can be adjusted through the `AlgorithmInformation` object.

* `name` is the display name in ADEPT
* `deep` is used for grouping the algorithms based on their type
* `explainable` is used to denote the support for advanced explainability techniques

#### Configuration Options

The configuration options that are exposed to the user can be adjusted through the `AlgorithmConfig` object.
ADEPT supports multiple different types of settings:

* Numeric (Integer, Float)
* Slider
* Toggle
* Dropdown

There are a few rules for using algorithm configs:

* The id's of the settings need to be unique
* Dropdown settings need to contain at least one Option
* Option names must be unique
* Options can include further settings (Numeric, Slider, Toggle) but no further dropdowns
* The default values have to be in the specified range
* The step value has to be greater than zero

An example for no exposed settings:

```python
config = AlgorithmConfig()
```

An example for a simple toggle setting:

```python
toggle_setting = ToggleSetting(id="toggle",
                               name="A simple toggle setting",
                               description="This is a description.",
                               default=False)

config = AlgorithmConfig([toggle_setting])
```

An example for multiple settings:

```python
toggle_setting = ToggleSetting(id="toggle",
                               name="A simple toggle setting",
                               description="This is a description.",
                               default=False)

float_setting = FloatSetting(id="float",
                             name="A float setting",
                             description="This is a description.",
                             default=3.14)

slider_setting = SliderSetting(id="slider",
                               name="A slider setting",
                               description="This is a description.",
                               default=42,
                               step=0.5,
                               lowBound=0,
                               highBound=100)

config = AlgorithmConfig([toggle_setting, float_setting, slider_setting])
```

An example for a simple dropdown:

```python
dropdown = OptionSetting(id="dropdown",
                         name="Language",
                         description="Select the desired language.",
                         default="English",
                         options=[Option("English"), Option("German")])

config = AlgorithmConfig([dropdown])
```

An example for a dropdown with settings for options:

```python
toggle_setting = ToggleSetting(id="toggle",
                               name="A simple toggle setting",
                               description="This is a description.",
                               default=False)

float_setting = FloatSetting(id="float",
                             name="A float setting",
                             description="This is a description.",
                             default=3.14)

option_a = Option(name="Toggle Option", settings=[toggle_setting])

option_b = Option(name="Float Option", settings=[float_setting])

dropdown = OptionSetting(id="dropdown",
                         name="Dropdown",
                         description="This is a description.",
                         default="Toggle Option",
                         options=[option_a, option_b])

config = AlgorithmConfig([dropdown])
```

Copyright © ADEPT ML, TU Dortmund 2022
