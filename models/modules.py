import torch

# Defines a custom neural network module that transforms a latent vector z into 
# a tensor with shape suitable for use as a neural CDE vector field.
# The output is squashed with a tanh non-linearity to bound the dynamics.
class FinalTanh(torch.nn.Module):
    # Initializes the module:
    # - linear_in maps from hidden_channels to hidden_hidden_channels
    # - linears is a stack of (num_hidden_layers - 1) hidden layers with ReLU activation
    # - linear_out maps to a tensor of shape (hidden_channels × input_channels)
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(
            hidden_channels, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(
            hidden_hidden_channels, input_channels * hidden_channels)
    # Returns a string representation of the model's configuration for debugging/logging
    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels,
                         self.hidden_hidden_channels, self.num_hidden_layers)
    # Applies the sequence of linear + ReLU layers, then reshapes the output
    # The final tanh activation bounds the output values between [-1, 1]
    # Output shape: (*batch_shape, hidden_channels, input_channels)
    def forward(self, z):

        z = self.linear_in(z)
        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(
            *z.shape[:-1], self.hidden_channels, self.input_channels)
        z = z.tanh()
        return z