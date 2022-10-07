from typing import Any, Dict, Union, TypeVar
from .template import Input, Output
from PIL.Image import Image

Tensor = TypeVar("Tensor")

class Inference:
    '''An interface abstracting the invocation lifecycle of any ML model.

    All the work in the future is made possible based on an assumption that the
    invocation of any model can be divided into five steps that make up a lifecycle
    which looks like

                    +----------------------------+ on receiving an input
                    v                            |
        load --> pre-process --> forward --> post-process --> unload


    1. load: Loads the model into memory.
    2. pre-process: Converts raw input data into a framework-specific tensor.
    3. forward: Call the model using the tensor given by pre-process
    4. post-process: Convert the output given by forward into human-readable formats.
    5. unload: Drops the loaded model from memory.

    Apparently, these steps can be implemented as five methods in a class whereas
    not only can inferencing invocation be made simple and standarized, but also
    the deployment of models no longer be a pain. Because by doing so users does
    not need to know what the model they are going to use looks like or how it is
    implemented. The model is just a blackbox to its users.

    Therefore, you need to define a class that inherits this class and implement
    the five methods below by embedding your code in them.
    '''

    def __init__(self, path: str) -> None:
        '''Loads the model from the given path into memory.

        To load the model into memory, this method must receive the path to your
        saved model. A method named "load" is typically available within the library
        that was used to save the model. You don't have to implement this method
        if you don't have such a saved model.

        Args:
        path: A string representing the path to a model.

        Raises:
        NotImplementedError: An error occurred when this method is not implemented
            by subclasses.
        '''
        raise NotImplementedError

    def __del__(self) -> None:
        '''Drops the loaded model from memory.

        On deleting an Inference object, this method will be called and the loaded 
        model need to be dropped from memory. So you must manually free out the
        memories taken by your model for implementing this method.

        Raises:
        NotImplementedError: An error occurred when this method is not implemented
            by subclasses.
        '''

    def preprocess(self, input: Input) -> Tensor:
        '''Pre-processes an input.

        A union typed argument containing an input data is passed to this method
        to be converted into a framework-specific tensor. You can somewhat normalize
        the input during processing to satisfy your model's input dimension. Then
        return the tensor that will then be passed to the next method named "forward".
        Note that what you do in this method is typically the same as that in your
        training code.

        Args:
        input: A bytesarray object containing the input data.

        Returns: A framework-specific tensor object

        Raises:
        NotImplementedError: An error occurred when this method is not implemented
            by subclasses.
        '''
        raise NotImplementedError

    def forward(self, input: Tensor, **kargs) -> Tensor:
        '''Calls the model to make an inference.

        The tensor returned by method preprocess is passed to this method, in
        which you make a inferencing call to your model and return the output
        tensor that will be passed to next method named "postprocess".


        Args:
        input: A framework-specific tensor returned by method preprocess.
        kargs: A dict containing some options like
            onnx_input_name: A string specifying a ONNX input name

        Returns: A framework-specific tensor returned by your model.

        Raises:
        NotImplementedError: An error occurred when this method is not implemented
            by subclasses.
        '''
        raise NotImplementedError

    def postprocess(self, output: Tensor) -> Union[str, Dict[str, Any]]:
        '''Post-processes an output.

        The output returned by method forward is passed to this method to be
        converted into something that could be read and understood by human
        beings, the very species of the end users :)

        Args:
        output: A framework-specific tensor returned by your model.

        Returns: It returns a string converted from the output tenser if your
            model processes NLP tasks, otherwise a dict explaining each element
            in the output tensor. Anyways, the returned data must be human-readable
            other than raw numbers understood only by machines.

        Raises:
        NotImplementedError: An error occurred when this method is not implemented
            by subclasses.
        '''
        raise NotImplementedError


    def __call__(self, input: Input) -> Output:
        x = self.preprocess(input)
        y = self.forward(x)
        return self.postprocess(y)




