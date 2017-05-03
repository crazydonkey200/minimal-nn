import numpy as np
import functools


def check_backprop(model, data, labels, d=1e-6):
    "Check backprop implementation."
    original_params_vec = get_params_vec(model)
    f = functools.partial(loss, model=model, data=data, labels=labels)
    grad = functools.partial(backprop_grad, model=model, labels=labels)
    try:
        result = check_grad(f, grad, original_params_vec, d=d)
    finally:
        use_params_vec(original_params_vec, model)

    return result


def loss(params_vec, model, data, labels):
    "Compute loss with given parameter vector."
    use_params_vec(params_vec, model)
    probs, loss = model.forwardprop(data, labels)
    return loss


def backprop_grad(params_vec, model, labels):
    use_params_vec(params_vec, model)
    model.backprop(labels)
    return get_grad_params_vec(model)
    

def compute_fd_grad(f, params_vec, d):
    fd_grad = np.zeros_like(params_vec)
    for i in xrange(params_vec.shape[0]):
        d_v = np.zeros_like(params_vec)
        d_v[i] = d
        fd_grad[i] = (f(params_vec+d_v) - f(params_vec-d_v)) / (2*d)
    return fd_grad


def check_grad(f, grad, params_vec, d=1e-6):
    fd_grad = compute_fd_grad(f, params_vec, d)
    computed_grad = grad(params_vec)

    return (np.linalg.norm(fd_grad - computed_grad)
            / max(np.linalg.norm(fd_grad),
                  np.linalg.norm(computed_grad)))


def get_params_vec(model):
    return np.hstack([layer.w.flatten() for layer in model.layers]
                     + [layer.b.flatten() for layer in model.layers])


def get_grad_params_vec(model):
    result = np.hstack([layer.d_w.flatten() for layer in model.layers]
                       + [layer.d_b.flatten() for layer in model.layers])
    return result


def use_params_vec(params_vec, model):
    for layer, (w, b) in zip(model.layers, get_w_b_list(params_vec, model)):
        layer.w = w
        layer.b = b


def get_w_b_list(params_vec, model):
    w_list = []
    pointer = 0
    for layer in model.layers:
        shape = layer.w.shape
        flat_w = params_vec[pointer:pointer + shape[0] * shape[1]]
        w_list.append(flat_w.reshape(shape))
        pointer += shape[0] * shape[1]
        
    b_list = []
    for layer in model.layers:
        shape = layer.b.shape
        flat_w = params_vec[pointer:pointer + shape[1]]
        b_list.append(flat_w.reshape(shape))
        pointer += shape[0] * shape[1]
        
    return zip(w_list, b_list)
