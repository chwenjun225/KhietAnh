"""Defines base class SequenceModule, a modular interface for sequence models."""
from torch import nn
import functools

class SequenceModule(nn.Module):
    """Abstract sequence model class. All models must adhere to this interface.

    A SequenceModule is generally a model that transforms an input of shape
    (n_batch, l_sequence, d_model) to (n_batch, l_sequence, d_output)

    REQUIRED methods and attributes
    forward, d_model, d_output: controls standard forward pass, a sequence-to-sequence transformation
    __init__ should also satisfy the following interface; see SequenceIdentity for an example
        def __init__(self, d_model, transposed=False, **kwargs)

    OPTIONAL methods
    default_state, step: allows stepping the model recurrently with a hidden state
    state_to_tensor, d_state: allows decoding from hidden state
    """

    @property
    def d_model(self):
        """Model dimension (generally same as input dimension).

        This attribute is required for all SequenceModule instantiations.
        It is used by the rest of the pipeline (e.g. model backbone, encoder) 
        to track the internal shapes of the full model.
        """
        if getattr(self, "_d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_model")
        return self._d_model

    @d_model.setter
    def d_model(self, d):
        self._d_model = d

    @property
    def d_output(self):
        """Output dimension of model.

        This attribute is required for all SequenceModule instantiations.
        It is used by the rest of the pipeline (e.g. model backbone, decoder) 
        to track the internal shapes of the full model.
        """
        if getattr(self, "_d_output", None) is None:
            raise NotImplementedError("SequenceModule instantiation must specify d_output for decoder")
        return self._d_output

    @d_output.setter
    def d_output(self, d):
        self._d_output = d

    def forward(self, x, state=None, **kwargs):
        """Forward pass of sequence model, a sequence-to-sequence 
        transformation with an optional state.

        Generally, this should map a tensor of 
        shape (batch, length, self.d_model) to (batch, length, self.d_output)

        Additionally, it returns a "state" which can be any additional information
        For example, RNN and SSM layers may return their hidden state,
        while some types of transformer layers (e.g. Transformer-XL) 
        may want to pass a state as well.
        """
        return x, None
    
    






























































