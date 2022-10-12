## Installation

KubeMo needs Python (>= 3.7) installed on your device.

**From PyPi**

KubeMo has been published to [PyPi](https://pypi.org/), so you can install it simply by using pip.

```
pip install kubemo
```

Note that KubeMo is not ready for production until the first stable release, so you can eithor wait or help us reach the day earlier :)

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

**For developers**

If you would like to join us in improving KubeMo, please follow the instructions below.

```
pip install build twine

pip install -e .
```

Or simply use *make* if possible.

```
make develop
```

The development workflow is based on https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html