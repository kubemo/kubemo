from abc import abstractmethod
from typing import Optional, Tuple, Generic, TypeVar
from .serialize import Input, Output

Tensor = TypeVar('Tensor')


class Inference(Generic[Tensor]):
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

    def __init__(self,
            device_id: Optional[int] = None,
            input_names: Optional[Tuple[str, ...]] = None, 
            output_names: Optional[Tuple[str, ...]] = None,
        ) -> None:
        '''Loads the model from the given path into memory.

        To load the model into memory, this method must receive the path to your
        saved model. A method named "load" is typically available within the library
        that was used to save the model. You don't have to implement this method
        if you don't have such a saved model.

        Args:
        input_names: A tuple of strings each of which names an input of your model.
        output_names: A tuple of strings each of which names an output of your model.
        '''
        self.input_names = input_names
        self.output_names = output_names
        self.device_id = device_id
       

    def __del__(self) -> None:
        '''Drops the loaded model from memory.

        On deleting an Inference object, this method will be called and the loaded 
        model need to be dropped from memory. So you must manually free out the
        memories taken by your model for implementing this method.
        '''
        pass


    @abstractmethod
    def preprocess(self, inputs: Tuple[Input, ...]) -> Tuple[Tensor, ...]:
        '''Pre-processes an input.

        A list of dynamic Input objects each of which contains a single input data 
        is passed to this method to be converted into a framework-specific tensor. 
        You can somewhat normalize the input during processing to satisfy your model's
        input dimension. Then return the tensor that will then be passed to the next
        method named "forward". NOTE that what you do in this method is typically 
        the same as that in your training code.

        Args:
        inputs: A tuple of dynamic Input objects each of which contains a single input data.

        Returns: A tuple of framework-specific tensors converted from inputs.

        Raises:
        NotImplementedError: An error occurred when this method is not implemented
            by subclasses.
        '''
        raise NotImplementedError


    @abstractmethod
    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        '''Calls the model to make an inference.

        The tensor returned by method preprocess is passed to this method, in which
        you make a inferencing call to your model and return the output tensor that
        will be passed to next method named "postprocess".


        Args:
        inputs: A tuple of framework-specific tensors returned by method preprocess.

        Returns: A tuple of framework-specific tensor returned by your model.

        Raises:
        NotImplementedError: An error occurred when this method is not implemented
            by subclasses.
        '''
        raise NotImplementedError


    @abstractmethod
    def postprocess(self, outputs: Tuple[Tensor, ...]) -> Tuple[Output, ...]:
        '''Post-processes an output.

        The output returned by method forward is passed to this method to be converted
        into something that could be read and understood by human beings, the very 
        species of the end users :)

        Args:
        outputs: A tuple of framework-specific tensor returned by method forward.

        Returns: A tuple of Output objects converted from outputs.

        Raises:
        NotImplementedError: An error occurred when this method is not implemented
            by subclasses.
        '''
        raise NotImplementedError


    @abstractmethod
    def concat(self, batch: Tuple[Tuple[Tensor, ...], ...]) -> Tuple[Tensor, ...]:
        '''Concatenates a batch of inputs.

        Args:
        batch: A tuple of tuples each of which specifies a list of framework-specific 
            tensor returned by method preprocess.

        Returns: A tuple of framework-specific tensors respectively concatenated by
            the given batch

        Raises:
        NotImplementedError: An error occurred when this method is not implemented
            by subclasses.
        '''
        raise NotImplementedError


    def __call__(self, *args: Tuple[Input, ...]) -> Tuple[Tuple[Output, ...], ...]:
        batch_x = tuple(self.preprocess(x) for x in args)
        batch_y = self.forward(self.concat(batch_x))
        return tuple(self.postprocess(y) for y in zip(*batch_y))
