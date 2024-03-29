from torch.nn.utils.clip_grad import clip_grad_norm_

from auxilearn.implicit_diff import Hypergrad

# ----------------------------------------------------------------------

class MetaOptimizer:
    def __init__(self, meta_optimizer, hpo_lr, truncate_iter=3, max_grad_norm=10):
        """Auxiliary parameters optimizer wrapper

        :param meta_optimizer: optimizer for auxiliary parameters
        :param hpo_lr: learning rate to scale the terms in the Neumann series
        :param truncate_iter: number of terms in the Neumann series
        :param max_grad_norm: max norm for grad clipping
        """
        self.meta_optimizer = meta_optimizer
        self.hypergrad = Hypergrad(learning_rate=hpo_lr, truncate_iter=truncate_iter)
        self.max_grad_norm = max_grad_norm

    def step(self, train_loss, main_loss, primary_param, aux_params, return_grads=False, take_step=True):
        """

        :param train_loss: train loader
        :param val_loss:
        :param parameters: parameters (main net)
        :param aux_params: auxiliary parameters
        :param return_grads: whether to return gradients
        :return:
        """
        # zero grad
        self.zero_grad()
        
        # Get the hypergradients (d_main_loss/d_aux_params) times the hyperlearning rate
        hyper_gards = self.hypergrad.grad(
            main_loss=main_loss,
            train_loss=train_loss,
            aux_params=aux_params,
            primary_param=primary_param,
        )

        # meta step
        if take_step:
            self.optimizer_step(aux_params, hyper_gards)
            
        return hyper_gards if return_grads else None

    def zero_grad(self):
        self.meta_optimizer.zero_grad()
    

    def optimizer_step(self, aux_params, hyper_gards):
        for p, g in zip(aux_params, hyper_gards, strict=False):
            p.grad = g

        # grad clipping
        if self.max_grad_norm is not None:
            clip_grad_norm_(aux_params, max_norm=self.max_grad_norm)
        
        self.meta_optimizer.step()

# ----------------------------------------------------------------------
