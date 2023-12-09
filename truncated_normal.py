import jax.numpy as jnp
from jax.scipy.special import erf
import jax

@jax.jit
def mult(x, y):
    return jnp.nan_to_num(x * y, nan=0, posinf=jnp.inf, neginf=-jnp.inf)

@jax.jit
def beta_f(b, m=0, s=1):
    return (b - m)/s

@jax.jit
def alpha_f(a, m=0, s=1):
    return (a - m)/s

@jax.jit
def phi(x):
    return 1 / jnp.sqrt(2*jnp.pi) * jnp.exp(-jnp.square(x)/2)

@jax.jit
def big_phi(x, m=0, s=1):
    return 1 / 2 * (1 + erf((x - m)/(jnp.sqrt(2)*s)))

@jax.jit
def z(m, s, a=-jnp.inf, b=jnp.inf):
    return jnp.sqrt(2*jnp.pi)*s*(big_phi(b, m, s) - big_phi(a, m, s))

@jax.jit
def mu_f(m, s, a=-jnp.inf, b=jnp.inf):
    alpha, beta = alpha_f(a, m, s), beta_f(b, m, s)
    return m - s * (phi(beta) - phi(alpha)) / (big_phi(beta) - big_phi(alpha))

@jax.jit
def var_f(m, s, a=-jnp.inf, b=jnp.inf):
    alpha, beta = alpha_f(a, m, s), beta_f(b, m, s)
    return jnp.square(s) * (1 - (mult(beta, phi(beta)) - mult(alpha, phi(alpha))) / (big_phi(beta) - big_phi(alpha)) - jnp.square((phi(beta) - phi(alpha)) / (big_phi(beta) - big_phi(alpha))))

@jax.jit
def eta_1(m, s, a=-jnp.inf, b=jnp.inf):
    return mu_f(m, s, a, b)

@jax.jit
def eta_2(m, s, a=-jnp.inf, b=jnp.inf):
    return var_f(m, s, a, b) + jnp.square(mu_f(m, s, a, b))

@jax.jit
def tn_entropy(m, s, a=-jnp.inf, b=jnp.inf):
    alpha, beta = alpha_f(a, m, s), beta_f(b, m, s)
    return jnp.log(jnp.sqrt(2*jnp.pi*jnp.e)*s*(big_phi(beta) - big_phi(alpha))) + (mult(alpha, phi(alpha)) - mult(beta, phi(beta))) / (2 * (big_phi(beta) - big_phi(alpha)))

@jax.jit
def tn_kl(m1, s1, a1, b1, m2, s2, a2, b2):
    return (
        jnp.square(m2) / (2*jnp.square(s2))
      - jnp.square(m1) / (2*jnp.square(s1))
      + jnp.log(z(m2, s2, a2, b2))
      - jnp.log(z(m1, s1, a1, b1))
      - (m2 / jnp.square(s2) - m1 / jnp.square(s1))*(eta_1(m1, s1, a1, b1))
      - (1 / (2*jnp.square(s1)) - 1 / (2*jnp.square(s2)))*eta_2(m1, s1, a1, b1)
    )