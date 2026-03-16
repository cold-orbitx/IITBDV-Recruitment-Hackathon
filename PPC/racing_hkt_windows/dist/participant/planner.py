import numpy as np

def smooth_path(points, iterations=5):

    pts = points.copy()
    n = len(pts)

    for _ in range(iterations):

        new_pts = []

        for i in range(n):

            prev = pts[i-1]
            curr = pts[i]
            nextp = pts[(i+1) % n]

            new_pts.append((prev + curr + nextp) / 3)

        pts = np.array(new_pts)

    return pts


def interpolate_path(points, factor=2):

    dense = []
    n = len(points)

    for i in range(n):

        p0 = points[i-1]
        p1 = points[i]
        p2 = points[(i+1) % n]
        p3 = points[(i+2) % n]

        for t in np.linspace(0, 1, factor, endpoint=False):

            t2 = t * t
            t3 = t2 * t

            p = 0.5 * (
                (2*p1)
                + (-p0 + p2)*t
                + (2*p0 - 5*p1 + 4*p2 - p3)*t2
                + (-p0 + 3*p1 - 3*p2 + p3)*t3
            )

            dense.append(p)

    return np.array(dense)


def plan(cones):

    blue = []
    yellow = []

    for c in cones:

        if c["side"] == "left":
            blue.append(np.array([c["x"], c["y"]]))

        elif c["side"] == "right":
            yellow.append(np.array([c["x"], c["y"]]))

    blue = np.array(blue)
    yellow = np.array(yellow)

    midpoints = []

    for b in blue:

        dists = np.linalg.norm(yellow - b, axis=1)
        j = np.argmin(dists)

        y = yellow[j]

        mid = (b + y) / 2
        midpoints.append(mid)

    midpoints = np.array(midpoints)

    # smooth track
    midpoints = smooth_path(midpoints, iterations=3)

    # densify
    midpoints = interpolate_path(midpoints, factor=5)

    path = []

    for p in midpoints:

        path.append({
            "x": float(p[0]),
            "y": float(p[1])
        })

    return path