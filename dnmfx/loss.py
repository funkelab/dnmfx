from .utils import sigmoid
import jax
import jax.numpy as jnp


def l2_loss(H_logits, W_logits, B_logits, X):

    assert len(H_logits.shape) == 2
    assert len(W_logits.shape) == 2
    assert len(B_logits.shape) == 2

    # convert H, W, and B to positive values
    H = sigmoid(H_logits)
    W = sigmoid(W_logits)
    B = sigmoid(B_logits)

    # compute X_hat[t,s] = W[t,1]@H[1,s] + 1[t,1]@B[1,s]
    X_hat = W@H + B

    return jnp.linalg.norm(X - X_hat)


l2_loss_grad = jax.value_and_grad(l2_loss, argnums=(0, 1, 2))
