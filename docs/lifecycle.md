## Lifecycle

KubeMo is built based on an assumption that the invocation of any model can be divided into five steps that make up a lifecycle which looks like:

```
             +---------------------------+ on receiving a batch of inputs
             v                           |
load --> preprocess --> forward --> postprocess --> unload
```
Each node in the graph means an action and is explained as follows.

- **load** loads the model into memory.
- **preprocess** converts raw input data into a framework-specific tensor.
- **forward** call the model using the tensor given by *preprocess*
- **postprocess** convert the output given by *forward* into human-readable formats.
- **unload** drops the loaded model from memory.

Apparently, these steps can be somehow implemented as five methods in a class so that model invocation can be made simple and standarized whereas model deployment can no longer be painful for researchers since it is not requred to know what the model to be invoked or deployed looks like or how it is implemented. Any model should always be a blackbox to its users.

That is why KubeMo has defined a generic class named *Inference*, the one from which your self-defined class derives. So you need to implement its abstract methods by filling them with your model-specific code.

For example,

```python
from typing import Tuple
from kubemo import Inference, Input, Output
from numpy import ndarray, concatenate

class MyInference(Inference[ndarray]):

    def __init__(self,
        device_id: Optional[int] = None,
        input_names: Optional[Tuple[str, ...]] = None, 
        output_names: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__(input_names, output_names)

    def __del__(self): pass

    def preprocess(self, inputs: Tuple[Input, ...]) -> Tuple[ndarray, ...]: pass

    def forward(self, inputs: Tuple[ndarray, ...]) -> Tuple[ndarray, ...]: pass

    def postprocess(self, outputs: Tuple[ndarray, ...]) -> Tuple[Output, ...]: pass

    def concat(self, batch: Tuple[Tuple[ndarray, ...], ...]) -> Tuple[ndarray, ...]:
        return tuple(concatenate(x) for x in zip(*batch))
```

Relax, bro :) 

The first 5 methods are what we have just talked about, the lifecyle of the model to be invoked. Except that we replaced *load* and *unload* respectively with Pythonoic names as we know it *\_\_init__* and *\_\_del__* so that the model embedded in this class can be loaded on instantiation and dropped by Python's keyword *del*.

Note that the argument passed to *preprocess* or *postprocess* is a one-dimmensional tuple of objects - in this case *Input*s or *ndarray*s - which means all models are considered to have multiple inputs and multiple outputs so as to make KubeMo compatible with as many models as possible. So the argument passed to *preprocess* or *postprocess* is just a tuple with only one element if it is a model with single input or single output.

Wait! you may be wondering. How's KubeMo supposed to handle a batch invocation if it is a one-dimensional tuple of inputs passed to *preprocess*?

Well, it does receive a batch of input sets, but you don't have to know it, because KubeMo passes each set of inputs out of the batch to *preprocess* and concatenates them - by using the 6th method named *concat* - back to a one-dimmensional tuple of framework-specific tensors that will then be passed to *forward*. So does it for *postprocess* but reversely.

However, it may get tricky if you implement those methods yourselves. So KubeMo has implemented some [adaptors](https://github.com/kubemo/kubemo/tree/main/kubemo/adaptor) for the frameworks of today. With those adaptors, you implement no more methods than just *preprocess* and *postprocess*. Checkout the [examples](https://github.com/kubemo/kubemo/tree/main/example) we provide to see how to use those adaptors.
