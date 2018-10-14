"""
Lunar lander dynamics without a simulator
"""

import numpy as np

from redax.utils.overapprox import maxmincos, maxminsin


thrust_mult = 1.0
side_mult = 1.0

"""
Lander Parameters with VIEWPORT_W = 2400, VIEWPORT_H = 1600
"""
# vxtox = 0.0006301
# vytoy = .0014062
# costovx = .04236 # side thrust
# sintovy = .02826 # side thrust
# sintovx = -.2159 # vertical thrust
# costovy = .1439 # vertical thrust
# g = .1066

"""
Lander Parameters with VIEWPORT_W = 600, VIEWPORT_H = 400

Multiplies positional change coefficients by 16 from previous. 
Divides velocity change coefficients by 4 from previous.
Previous means VIEWPORT_W = 2400, VIEWPORT_H = 1600
"""
vxtox = .01031
vytoy = .0225
costovx = .01059 # side thrust
sintovy = .007067 # side thrust
sintovx = -.05397 # vertical thrust
costovy = 0.0359 # vertical thrust
omegatotheta = .05
sidetoomega = .05598
g = .026667


def lander_dynamics(x, y, vx, vy, theta, omega, a, discrete=True):
    
    thrust = 0.0
    side = 0.0
    if discrete:
        thrust = 1 if a == 2 else 0
        thrust *= thrust_mult
        side = a - 2 if a in [1, 3] else 0
    else:
        if a[0] > 0.0:
            thrust = (np.clip(a[0], 0.0,1.0) + 1.0)*0.5  # remap interval to [.5, 1.0]
        if np.abs(a[1]) > 0.5:
            direction = np.sign(a[1])
            side = direction * np.clip(np.abs(a[1]), 0.5,1.0)

    cos = np.cos(theta)
    sin = np.sin(theta)

    x_ = x + vxtox * vx
    y_ = y + vytoy * vy
    vx_ = vx + thrust * sintovx * sin + side * costovx * cos
    vy_ = vy + thrust * costovy * cos + side * sintovy * sin - g
    theta_ = theta + omegatotheta * omega
    omega_ = omega - side * sidetoomega

    return (x_, y_, vx_, vy_, theta_, omega_)

def lander_box_dynamics(x, y, vx, vy, theta, omega, a, steps, discrete=True):
    r"""
    Takes state boxes and computes an output box
    """

    for _ in range(1, steps + 1):


        thrust = 0.0
        side = 0.0
        if discrete:
            thrust = 1 if a == 2 else 0
            thrust *= thrust_mult
            side = a - 2 if a in [1, 3] else 0
        else:
            if a[0] > 0.0:
                thrust = (np.clip(a[0], 0.0,1.0) + 1.0)*0.5
            if np.abs(a[1]) > 0.5:
                direction = np.sign(a[1])
                side = direction * np.clip(np.abs(a[1]), 0.5,1.0)

        x_ = (x[0] + vxtox * vx[0], x[1] + vxtox * vx[1])
        y_ = (y[0] + vytoy * vy[0], y[1] + vytoy * vy[1])
        theta_ = (theta[0] + omegatotheta * omega[0], theta[1] + omegatotheta * omega[1])
        omega_ = (omega[0] - side * sidetoomega, omega[1] - side * sidetoomega)

        mincos, maxcos = maxmincos(theta[0], theta[1])
        minsin, maxsin = maxminsin(theta[0], theta[1])

        if side > .1:
            vx_ = (vx[0] + thrust * sintovx * maxsin + side * costovx * mincos,
                   vx[1] + thrust * sintovx * minsin + side * costovx * maxcos)
            vy_ = (vy[0] + thrust *  costovy * mincos + side * sintovy * minsin - g,
                   vy[1] + thrust *  costovy * maxcos + side * sintovy * maxsin - g)
        else: # side = -1 or 0
            vx_ = (vx[0] + thrust * sintovx * maxsin + side * costovx * maxcos,
                   vx[1] + thrust * sintovx * minsin + side * costovx * mincos)
            vy_ = (vy[0] + thrust *  costovy * mincos + side * sintovy * maxsin - g,
                   vy[1] + thrust *  costovy * maxcos + side * sintovy * minsin - g)

        x, y, vx, vy, theta, omega = x_, y_, vx_, vy_, theta_, omega_

    x, y, vx, vy, theta, omega = tuple((a[0], a[1]) for a in (x, y, vx, vy, theta, omega))

    return x, y, vx, vy, theta, omega

def plot_io_bounds(x, y, vx, vy, theta, omega, a, steps, discrete=True):
    import matplotlib.pyplot as plt

    statebox = [x, y, vx, vy, theta, omega]
    centerstate = [box[0] + .5*(box[1] - box[0]) for box in statebox]
    envstate = [i for i in centerstate]

    # Zero order hold on actions if needed
    if discrete and isinstance(a, int):
        a = a * np.ones(steps, dtype=np.int32)
    elif not discrete:
        a = [np.array(a) for i in range(steps)]

    # System IDed model trajectory
    centerstatehist = [centerstate]
    for i in range(steps):
        centerstate = lander_dynamics(*centerstate, a=a[i], discrete=discrete)
        centerstatehist.append(centerstate)

    # Actual openai gym model trajectory
    envstatehist = [envstate]
    if discrete:
        from lunar_lander import LunarLander
        env = LunarLander()
    else:
        from lunar_lander import LunarLanderContinuous
        env = LunarLanderContinuous()
    s = env.reset(envstate)
    for i in range(steps):
        s, _, _, _ = env.step(a[i])
        envstatehist.append(s[0:6])


    # Overapproximated trajectory
    stateboxhist = [statebox]
    for i in range(steps):
        statebox = lander_box_dynamics(*statebox, a=a[i], steps=1, discrete=discrete)
        stateboxhist.append(statebox)

    centerstatehist = np.array(centerstatehist)
    envstatehist = np.array(envstatehist)
    stateboxhist = np.array(stateboxhist)

    t = np.linspace(0, steps, steps+1)
    fig, axs = plt.subplots(6,1, figsize=(4, 9))

    # fig.set_size_inches(5,7,forward=True)

    limits = [[-1,1], [0,1], [-1, 1], [-1,1], [-np.pi/3, np.pi/3], [-.5,.5]]
    for i in range(6):
        axs[i].fill_between(t, stateboxhist[:, i, 0], stateboxhist[:,i,1],alpha=0.3)
        axs[i].plot(centerstatehist[:,i], 'r')
        axs[i].plot(envstatehist[:,i], 'b.')
        axs[i].set_ylim(bottom=limits[i][0], top=limits[i][1])
        axs[i].set_yticks(np.linspace(limits[i][0], limits[i][1], 17), minor=True)
        axs[i].grid(which='minor', alpha = .4)

    axs[0].set_title('Action {0}'.format(a))
    plt.show()

