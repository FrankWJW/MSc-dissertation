import torch

class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model, target):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook = self.hook_layers()
        self.target = target
    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
#             print('hook called')
#             print(grad_in)
            self.gradients = grad_in[1]

        # Register hook to the first layer
        first_layer = list(self.model._modules.items())[0][1]
#         print(first_layer) 
        hook = first_layer.register_backward_hook(hook_function)
        return hook
    def generate_gradients(self, input_image):
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
#         one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().to('cuda:0')
#         one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=self.target)
#         hook.remove()
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
#         print(self.gradients)
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        self.hook.remove()
        return gradients_as_arr