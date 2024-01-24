import torch

# ----------------------------------------------------------------------

class Hypergrad:
    """Implicit differentiation for auxiliary parameters.
    This implementation follows the Algs. in "Optimizing Millions of Hyperparameters by Implicit Differentiation"
    (https://arxiv.org/pdf/1911.02590.pdf), with small differences.

    """

    def __init__(self, learning_rate=.1, truncate_iter=3):
        self.learning_rate = learning_rate
        self.truncate_iter = truncate_iter

    def grad(self, main_loss, train_loss, aux_params, primary_param):
        """Calculates the learning_rate times the gradient of the main loss at the optimal weights (W*) w.r.t. the auxiliary parameters (Phi).
        """
        #  Compute the gradients of the train loss (main + auxiliary) w.r.t. the primary model's parameters W
        d_train_loss_dw = torch.autograd.grad(train_loss, list(primary_param), create_graph=True, retain_graph=True)

        # compute the gradient of the main loss w.r.t. the primary model's parameters w at current fixed point weights (assumed to be approx W*)
        d_main_loss_dw = torch.autograd.grad(main_loss, list(primary_param), create_graph=True)

        # Compute approx to  lr * H^-1 @ d_main_loss_dw, where H is the Hessian of the train loss w.r.t. the primary model's parameters W.
        d_main_dw_times_inv_hess = approx_vector_prod_with_inv_hessian(
            vec=d_main_loss_dw,
            grad_w=d_train_loss_dw,
            params_w=list(primary_param),
            alpha=self.learning_rate,
        )

        # multiply by -1:
        d_main_dw_times_inv_hess = [-g for g in d_main_dw_times_inv_hess]

        # compute the Jacobian of d_train_loss_dw w.r.t. the auxiliary model's parameters (d/dPhi d_train_loss_dw) [n_W x n_Phi]
        # times the vector dmain_dw_times_inv_hess  [n_W x 1] to get the gradient of the main loss w.r.t. the auxiliary model's parameters Phi
        # at W* [n_Phi x 1]
        d_main_d_phi = torch.autograd.grad(
            d_train_loss_dw,
            list(aux_params),
            grad_outputs=d_main_dw_times_inv_hess,
            allow_unused=True,
            retain_graph=True,
        )
        
        return d_main_d_phi
                    
# ----------------------------------------------------------------------

def approx_vector_prod_with_inv_hessian(vec: torch.Tensor, grad_w, params_w, alpha=0.1, truncate_iter=4):
    """
    Approximates the vector product with the inverse Hessian matrix using the truncated sum approximation.
    The Hessian is H = d(grad_w)/d(params_w) [W x W] and the vector is vec [W x 1].
    The function approximates  inv(alpha * H) @ vec [1 x W] with:
    [sum_{i=0..K}  (1 - alpha * H)^i] @ vec
    where K is the number of iterations in the truncated sum approximation.
    Since the hessian is too large for memory, we re-compute the hessian times vector product at each iteration,
    using this form:
    [sum_{i=0..K}  c_i] @ vec
    where c_i = (1 - alpha * H)^i = (1 - alpha * H) @ c_{i-1} @ = c_{i-1} - alpha *  H @ c_{i-1}, c_0 = I
    thus the sum can be computed by:
    sum_{i=0..K} b_i, where b_i = c_i @ vec =  b_{i-1} - alpha *  H @ b_{i-1}, and b_0 = vec

    Args:
        vec: a vector of size W
        grad_w: the gradient of the loss w.r.t. the model's parameters
        params_w: the model's parameters
        alpha: a scalar
        truncate_iter:  # how many elements in the truncated sum that approximates the inverse Hessian vector product
    """


    b = vec  # the current summed vector (at i=0).
    s = vec  # the current accumalated sum (up to i=0).

    # multiply grad_w by alpha:
    grad_w = [g * alpha for g in grad_w]

    for _ in range(1, truncate_iter):
        # Compute the Hessian times vector product:  alpha * H @ b_{i-1}
        H_b = torch.autograd.grad(
            grad_w,
            params_w,
            grad_outputs=b,
            retain_graph=True,
            allow_unused=True,
        )

        # Update b_i = b_{i-1} - alpha *  H @ b_{i-1}
        b = [curr_b - curr_h for (curr_b, curr_h) in zip(b, H_b, strict=False)]

        # update the accumalated sum: s_i = s_{i-1} + b_i
        s = [curr_s + curr_b for (curr_s, curr_b) in zip(s, b, strict=False)]

    return s
# ----------------------------------------------------------------------

# class Hypergrad:
#     """Implicit differentiation for auxiliary parameters.
#     This implementation follows the Algs. in "Optimizing Millions of Hyperparameters by Implicit Differentiation"
#     (https://arxiv.org/pdf/1911.02590.pdf), with small differences.

#     """

#     def __init__(self, learning_rate=.1, truncate_iter=3):
#         self.learning_rate = learning_rate
#         self.truncate_iter = truncate_iter

#     def grad(self, loss_val, loss_train, aux_params, params):
#         """Calculates the gradients w.r.t \phi dloss_aux/dphi, see paper for details

#         :param loss_val:
#         :param loss_train:
#         :param aux_params:
#         :param params:
#         :return:
#         """
#         dloss_val_dparams = torch.autograd.grad(
#             loss_val,
#             params,
#             retain_graph=True,
#             allow_unused=True
#         )

#         dloss_train_dparams = torch.autograd.grad(
#                 loss_train,
#                 params,
#                 allow_unused=True,
#                 create_graph=True,
#         )

#         v2 = self._approx_inverse_hvp(dloss_val_dparams, dloss_train_dparams, params)

#         v3 = torch.autograd.grad(
#             dloss_train_dparams,
#             aux_params,
#             grad_outputs=v2,
#             allow_unused=True
#         )

#         # note we omit dL_v/d_lambda since it is zero in our settings
#         return list(-g for g in v3)

#     def _approx_inverse_hvp(self, dloss_val_dparams, dloss_train_dparams, params):
#         """

#         :param dloss_val_dparams: dL_val/dW
#         :param dloss_train_dparams: dL_train/dW
#         :param params: weights W
#         :return: dl_val/dW * dW/dphi
#         """
#         p = v = dloss_val_dparams

#         for _ in range(self.truncate_iter):
#             grad = torch.autograd.grad(
#                     dloss_train_dparams,
#                     params,
#                     grad_outputs=v,
#                     retain_graph=True,
#                     allow_unused=True
#                 )

#             grad = [g * self.learning_rate for g in grad]  # scale: this a is key for convergence

#             v = [curr_v - curr_g for (curr_v, curr_g) in zip(v, grad)]
#             # note: different than the pseudo code in the paper
#             p = [curr_p + curr_v for (curr_p, curr_v) in zip(p, v)]

#         return list(pp for pp in p)
