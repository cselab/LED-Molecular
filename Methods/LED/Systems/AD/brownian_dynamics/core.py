#! /usr/bin/env python

import numpy as np


class NoBoundaries:
    def __init__(self):
        pass

    def apply(self, pos):
        return pos


class ReflectiveBoundaries:
    def __init__(self, lo, up):
        self.lo = np.array(lo)
        self.up = np.array(up)
        self.L = up - lo

    def apply(self, pos):
        n = len(pos)
        extra_lo = -np.minimum(pos - self.lo, np.zeros(pos.shape))
        extra_up = +np.maximum(pos - self.up, np.zeros(pos.shape))

        def length_to_cover(x, l):
            x -= np.floor(0.5 / l * x) * 2 * l
            return (x > l) * (2 * l - x) + (x <= l) * x

        extra_lo = length_to_cover(extra_lo, self.L)
        extra_up = length_to_cover(extra_up, self.L)

        pos = np.minimum(self.up * np.ones(pos.shape), pos)
        pos = np.maximum(self.lo * np.ones(pos.shape), pos)

        pos += extra_lo
        pos -= extra_up
        return pos


class NoDrift:
    def __init__(self):
        pass

    def apply(self, pos, t, dt):
        return pos


class DriftCosine:
    def __init__(self, A, omega):
        self.omega = np.array(omega)
        self.A = np.array(A)
        assert (self.omega.shape == self.A.shape)

    def apply(self, pos, t, dt):
        return pos + dt * np.cos(self.omega * t) * self.A


class BrownianDynamics:
    def __init__(self, positions, D, dt, boundaries, advection=NoDrift()):
        self.positions = np.array(positions)
        self.D = D
        self.dt = dt
        # self.seed       = random_seed
        self.time = 0

        self.advection = advection
        self.boundaries = boundaries

        self.sqrt_Ddt = np.sqrt(D * dt)
        self.n = len(positions)

    def advance(self, nsteps):
        for step in range(nsteps):
            dx = np.random.normal(np.zeros(self.positions.shape),
                                  self.sqrt_Ddt)
            self.positions += dx
            self.positions = self.advection.apply(self.positions, self.time,
                                                  self.dt)
            self.positions = self.boundaries.apply(self.positions)
            self.time += self.dt
