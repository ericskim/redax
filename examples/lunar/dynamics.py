"""
Lunar lander dynamics without a simulator
"""

import numpy as np

from redax.utils.overapprox import maxmincos, maxminsin
from lunar_lander import LunarLander

def lander_dynamics(x, y, vx, vy, theta, omega, a):
    
    thrust = 1 if a == 2 else 0
    side = a - 2 if a in [1, 3] else 0

    cos = np.cos(theta)
    sin = np.sin(theta)

    x_ = x + 0.0006301 * vx
    y_ = y + .0014062 * vy
    vx_ = vx + thrust * -.2159 * sin + side * .04236 * cos
    vy_ = vy + thrust *  .1439 * cos + side * .02826 * sin - .1066
    theta_ = theta + .05 * omega
    omega_ = omega - side * .05598

    return (x_, y_, vx_, vy_, theta_, omega_)


def lander_box_dynamics(x, y, vx, vy, theta, omega, a, steps):
    r"""
    Takes state boxes and computes an output box
    """

    for _ in range(1, steps + 1):

        thrust = 1 if a == 2 else 0
        side = a - 2 if a in [1, 3] else 0

        x_ = (x[0] + 0.0006301 * vx[0], x[1] + 0.0006301 * vx[1])
        y_ = (y[0] + .0014062 * vy[0], y[1] + .0014062 * vy[1])
        theta_ = (theta[0] + .05 * omega[0], theta[1] + .05 * omega[1])
        omega_ = (omega[0] - side * .05598, omega[1] - side * .05598)

        mincos, maxcos = maxmincos(theta[0], theta[1])
        minsin, maxsin = maxminsin(theta[0], theta[1])

        if side == 1:
            vx_ = (vx[0] + thrust * -.2159 * maxsin + side * .04236 * mincos,
                   vx[1] + thrust * -.2159 * minsin + side * .04236 * maxcos)
            vy_ = (vy[0] + thrust *  .1439 * mincos + side * .02826 * minsin - .1066,
                   vy[1] + thrust *  .1439 * maxcos + side * .02826 * maxsin - .1066)
        else: # side = -1 or 0
            vx_ = (vx[0] + thrust * -.2159 * maxsin + side * .04236 * maxcos,
                   vx[1] + thrust * -.2159 * minsin + side * .04236 * mincos)
            vy_ = (vy[0] + thrust *  .1439 * mincos + side * .02826 * maxsin - .1066,
                   vy[1] + thrust *  .1439 * maxcos + side * .02826 * minsin - .1066)

        x, y, vx, vy, theta, omega = x_, y_, vx_, vy_, theta_, omega_

    x, y, vx, vy, theta, omega = tuple((a[0]-.01, a[1]+.01) for a in (x, y, vx, vy, theta, omega))

    return x, y, vx, vy, theta, omega

def plot_io_bounds(x, y, vx, vy, theta, omega, a, steps):
    import matplotlib.pyplot as plt


    statebox = [x, y, vx, vy, theta, omega]
    centerstate = [box[0] + .5*(box[1] - box[0]) for box in statebox]
    envstate = [i for i in centerstate]

    if isinstance(a, int):
        a = a * np.ones(steps, dtype=np.int32)

    # System IDed model trajectory
    centerstatehist = [centerstate]
    for i in range(steps):
        centerstate = lander_dynamics(*centerstate, a=a[i])
        centerstatehist.append(centerstate)

    # Actual openai gym model trajectory
    envstatehist = [envstate]
    env = LunarLander()
    s = env.reset(envstate)
    for i in range(steps):
        s, _, _, _ = env.step(a[i])
        envstatehist.append(s[0:6])


    # Overapproximated trajectory
    stateboxhist = [statebox]
    for i in range(steps):
        statebox = lander_box_dynamics(*statebox, a=a[i], steps=1)
        stateboxhist.append(statebox)

    centerstatehist = np.array(centerstatehist)
    envstatehist = np.array(envstatehist)
    stateboxhist = np.array(stateboxhist)

    t = np.linspace(0, steps, steps+1)
    fig, axs = plt.subplots(6,1)


    limits = [[-1,1], [0,1], [-2.5, 2.5], [-5,3], [-np.pi/4, np.pi/4], [-1.5,1.5]]
    for i in range(6):
        axs[i].fill_between(t, stateboxhist[:, i, 0], stateboxhist[:,i,1],alpha=0.3)
        axs[i].plot(centerstatehist[:,i], 'r')
        axs[i].plot(envstatehist[:,i], 'b.')
        axs[i].set_ylim(bottom=limits[i][0], top=limits[i][1])

    axs[0].set_title('Action {0}'.format(a))
    plt.show()

