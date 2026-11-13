"""Anka's room, described once, in Anka's own coordinates.

This mirrors the virtual room in anka/body.py (FURNITURE, START, WALL_THINGS, LANDMARKS) so that the MuJoCo
room and the old 2D room are the same place. If you move something in body.py, move it here too and run
`python make_scene.py` to rebuild the MuJoCo scene.

Anka's coordinates (metres):  x goes east (towards the door), y goes north (towards the window); the room is
5 x 4 m. Heading 0 = east, 90 = north.

MuJoCo's duck always starts at the origin looking along +x, so the scene is shifted and turned so that
Anka's start spot (2.5, 1.4, looking north) is MuJoCo's origin:

    mj_x = ay - 1.4          (north of the start  ->  in front of the duck)
    mj_y = 1.6 - ax          (west of the start   ->  to the duck's left)
"""
import math

ROOM_W, ROOM_H = 5.0, 4.0
START = (1.6, 1.4, 90.0)          # where Anka starts (clear of the table and the sofa), and which way he looks (degrees, 90 = north)
WALL_HEIGHT = 2.5                 # a real room: with the camera 72 degrees tall (it stands on its side) he must not see sky over the walls

FURNITURE = {                     # name: (x1, y1, x2, y2, height in cm)
    "sofa": (0.3, 0.0, 2.3, 0.9, 80),
    "table": (2.6, 1.8, 3.6, 2.6, 75),
    "bookshelf": (4.6, 2.4, 5.0, 4.0, 180),
}
WALL_THINGS = {"window": (2.0, ROOM_H), "door": (ROOM_W, 1.3)}
THINGS = {                        # the small things (the ball is loose and can be kicked, the rest sit still)
    "ball": (3.8, 0.7),
    "sock": (1.0, 2.9),
    "guitar": (4.3, 3.7),
    "cup": (3.1, 2.2),            # stands on the table
}


def to_mj(ax, ay):
    """Anka's (x, y) -> MuJoCo's (x, y)."""
    return ay - START[1], START[0] - ax


def from_mj(mx, my):
    """MuJoCo's (x, y) -> Anka's (x, y)."""
    return START[0] - my, mx + START[1]


def heading_from_mj_yaw(yaw_rad):
    """MuJoCo/odometry yaw (0 = along the duck's start direction, positive = left) -> Anka's heading in degrees."""
    return (START[2] + math.degrees(yaw_rad)) % 360
