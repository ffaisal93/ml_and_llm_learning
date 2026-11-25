"""W160 executable answer; demonstration fixtures are explicitly labeled."""
import math

# uniform() is supplied by the caller/environment.
def sample_gaussian():
    u1 = uniform()
    u2 = uniform()
    radius = math.sqrt(-2.0 * math.log1p(-u1))
    angle = 2.0 * math.pi * u2
    return float(radius * math.cos(angle))
