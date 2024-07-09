# sigmaifGPT

![Python Version](https://img.shields.io/badge/python-3.8.1+-4584B6?logo=python)
![GitHub License](https://img.shields.io/github/license/jakub-galazka/sigma-if-gpt)

Modified [minGPT](https://github.com/karpathy/minGPT) model by Andrej Karpathy using Sigma-if context neurons with selective attention mechanism.

## Installation

First download this repository:

```bash
git fork https://github.com/jakub-galazka/sigma-if-gpt.git
```

Then create and activate a [virtual environment](https://docs.python.org/3/tutorial/venv.html) in the project directory:

```bash
python -m venv .venv
```

And install all the requirements from the file:

```bash
pip install -r requirements.txt
```

## Usage

Training of the model is based on the JSON configuration file. The `config_filepath` positional argument is required:

```bash
train.py config_filepath
```

Configuration file example:

```json
{
    "seed": 0,
    "dataset": {
        "sequenceLength": 6,
        "digitsNumber": 3
    },
    "model": {
        "architecture": "sigmaifGPT",
        "type": "gpt-nano",
        "parameters": {
            "K": 2,
            "aggregationThreshold": 0.6
        }
    }
}
```

## References

* Karpathy, A. [**A minimal PyTorch re-implementation of the OpenAI GPT (Generative Pretrained Transformer) training**](https://github.com/karpathy/minGPT).

* Huk, M. (2009) [**Learning distributed selective attention strategies with the Sigma-if neural network**](https://www.ii.pwr.edu.pl/~huk/open/INT_2009.pdf), in M. Akbar and D. Hussain (Eds.), Advances in Computer Science and IT, InTech, Vukovar, pp. 209-232.
