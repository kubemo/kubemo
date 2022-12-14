# KubeMo

KubeMo aims to simplify ML model deployment by aggregating as many machine-learning frameworks as possible into unified APIs along with the help of cloud-native toolkits.

Note that the APIs may vary since we are still in the POC stage.


## Installation

KubeMo needs Python (>= 3.7) installed on your device. Then you can install it either from [PyPi](https://pypi.org/) or manally.

**From PyPi**

```
pip install kubemo
```

Note that KubeMo is not ready for production until the first stable release if the POC works, so you can eithor wait or help us reach the day earlier :)

**Manually**

Clone or download this repo and install it by using pip in its root directory.

```
git clone https://github.com/kubemo/kubemo.git && cd kubemo

pip install .
```

Or add the `-e` flag if you would like to play around KubeMo before an actual installation.

```
pip install -e .
```

## Documentation

Check out [our documentation site](https://kubemo.github.io/kubemo) for detailed information about KubeMo, or you may first take a look at some [basic examples](./example/).

## Contribution

[Pull requests](https://github.com/kubemo/kubemo/pulls) are welcome. Note that every PR needs to refer to an issue, so please submit an issue before sending a new PR.

## Feedbacks

Bugs [here](https://github.com/kubemo/kubemo/issues) and ideas [here](https://github.com/kubemo/kubemo/discussions), thanks.

## License

KubeMo is Apache 2.0 licensed.