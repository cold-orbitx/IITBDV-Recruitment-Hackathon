import numpy as np

current_wp = 0
prev_steer = 0.0


def path_curvature(path, i):

    i1 = i
    i2 = (i + 3) % len(path)
    i3 = (i + 6) % len(path)

    p1 = np.array([path[i1]["x"], path[i1]["y"]])
    p2 = np.array([path[i2]["x"], path[i2]["y"]])
    p3 = np.array([path[i3]["x"], path[i3]["y"]])

    v1 = p2 - p1
    v2 = p3 - p2

    v1 /= np.linalg.norm(v1) + 1e-6
    v2 /= np.linalg.norm(v2) + 1e-6

    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    cross = v1[0]*v2[1] - v1[1]*v2[0]

    angle = np.arccos(dot)

    return angle * np.sign(cross)


def steering(path, state):

    global current_wp, prev_steer

    x = state["x"]
    y = state["y"]
    yaw = state["yaw"]
    speed = abs(state["vx"])

    lookahead = 1.1 + 0.22 * speed
    lookahead = np.clip(lookahead, 1.1, 3.0)

    while True:

        next_i = (current_wp + 1) % len(path)

        wp = path[next_i]

        dist = np.hypot(wp["x"] - x, wp["y"] - y)

        if dist < lookahead:
            current_wp = next_i
        else:
            break

    target = path[(current_wp + 1) % len(path)]

    dx = target["x"] - x
    dy = target["y"] - y

    local_x = np.cos(-yaw) * dx - np.sin(-yaw) * dy
    local_y = np.sin(-yaw) * dx + np.cos(-yaw) * dy

    L = np.hypot(local_x, local_y) + 1e-6

    curvature = 2 * local_y / (L * L)

    steer = curvature

    steer += 0.3 * path_curvature(path, current_wp)

    # faster steering response for S-corners
    max_delta = 0.065
    steer = np.clip(steer, prev_steer - max_delta, prev_steer + max_delta)

    prev_steer = steer

    return float(np.clip(steer, -0.40, 0.40))


def speed_control(path, state, steer):

    global current_wp

    speed = abs(state["vx"])

    curvature = abs(path_curvature(path, current_wp))

    # lateral grip model
    lat_accel = 10.5

    target_speed = np.sqrt(lat_accel / (curvature + 1e-3))

    # speed limits
    max_speed = 30.0
    min_speed = 10.0

    target_speed = np.clip(target_speed, min_speed, max_speed)

    # reduce speed when steering sharply
    steer_factor = 1 - min(abs(steer)/0.45, 1.0)
    target_speed *= 0.7 + 0.3 * steer_factor

    speed_error = target_speed - speed

    max_accel = 150
    max_decel = 35

    if speed_error > 0:
        accel = min(max_accel, 5 * speed_error)
    else:
        accel = -min(max_decel, -speed_error)

    throttle = np.clip(accel / max_accel, 0, 1)
    brake = np.clip(-accel / max_decel, 0, 1)

    return throttle, brake


def control(path, state, cmd_feedback, step):

    steer = steering(path, state)

    throttle, brake = speed_control(path, state, steer)

    return float(throttle), float(steer), float(brake)


