import numpy as np
import scipy.interpolate

from .utils import viewmatrix


def generate_interp_path(
    poses: np.ndarray,
    n_interp: int,
    spline_degree: int = 5,
    smoothness: float = 0.03,
    rot_weight: float = 0.1,
    endpoint: bool = False,
):
    """
    Creates a smooth spline path between input keyframe camera poses.

    Spline is calculated with poses in format (position, lookat-point, up-point).

    Args:
      poses: (n, 3, 4) array of input pose keyframes.
      n_interp: returned path will have n_interp * (n - 1) total poses.
      spline_degree: polynomial degree of B-spline.
      smoothness: parameter for spline smoothing, 0 forces exact interpolation.
      rot_weight: relative weighting of rotation/translation in spline solve.

    Returns:
      Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
    """
    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos    = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up     = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=endpoint)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(
        points, n_interp * (points.shape[0] - 1), k=spline_degree, s=smoothness
    )
    return points_to_poses(new_points)


def generate_spiral_path(
    poses, 
    bounds, 
    n_frames: int = 120, 
    n_rots: int = 2, 
    zrate: float = 0.5, 
    endpoint: bool = False, 
    radii=None,
):
    """Calculates a forward facing spiral path for rendering."""
    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of near and far bounds in disparity space.
    near_depth = bounds.min() * 0.9
    far_depth = bounds.max() * 5.0

    dt = 0.75
    focal = 1 / ((1 - dt) / near_depth + dt / far_depth)

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    if radii is None:
        radii = np.percentile(np.abs(positions), 90, 0)
    radii = np.concatenate([radii, [1.0]])

    # Generate poses for spiral path.
    render_poses = []
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    for theta in np.linspace(0.0, 2.0 * np.pi * n_rots, n_frames, endpoint=endpoint):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]
        position = cam2world @ t
        lookat = cam2world @ [0, 0, -focal, 1.0]
        z_axis = position - lookat
        render_poses.append(viewmatrix(z_axis, up, position))
    render_poses = np.stack(render_poses, axis=0)
    return render_poses


