"""Builds scene/anka_room.xml: Anka's room (see room_layout.py) as a MuJoCo scene for Pollen's duck simulator.

    python make_scene.py                       # uses ~/Pollen/microduck_rl
    python make_scene.py --rl /path/to/microduck_rl

It does NOT change anything in Pollen's repositories. The scene includes Pollen's own robot model by its full path
and gets the robot's mesh files through a small link (scene/assets) next to the scene file.
"""
import argparse
import os
from pathlib import Path

import room_layout as R

HERE = Path(__file__).resolve().parent
MODEL_SUBDIR = Path("src/mjlab_microduck/robot/microduck")

# The robot's own poses, copied from Pollen's scene.xml (the keyframes must be as long as the scene's state, and the
# ball adds 7 numbers, so they are written here again with the ball's place added at the end).
POSES = {
    "INIT": ("0 0 0.12 1 0 0 0  0 0 0 0 0 0 0 0 0 0 0 0 0 0",
             "0 0 0 0 0 0 0 0 0 0 0 0 0 0"),
    "STAND": ("0 0 0.12 1 0 0 0  0 -0.08726646259971647 -0.457924 -0.004940 0.452984 0.3490658503988659 "
              "0.3490658503988659 0 0  0 0.08726646259971647 0.457924 0.004940 -0.452984",
              "0 -0.08726646259971647 -0.457924 -0.004940 0.452984 0.3490658503988659 0.3490658503988659 0 0 "
              "0 0.08726646259971647 0.457924 0.004940 -0.452984"),
    "SIT": ("0 0 0.07 1 0 0 0  0 0 -0.5236 1.0472  0 0.5 1.6 0  0 0 0 0.5236 -1.0472 0",
            "0 0 -0.5236 1.0472 0 0.5 1.6 0 0 0 0 0.5236 -1.0472 0"),
    "FOLD": ("0 0 0.07 1 0 0 0  0 0 1.57 1.57 0  1 1 0 0  0 -1.57 -1.57 0",
             "0 0 1.57 1.57 0 1 1 0 0 0 0 -1.57 -1.57 0"),
}


# ---- what is in the room (change these and run Start Anka sim.command again) ----
ROOM_STYLE = "pollen"               # "pollen" = Pollen's own floor (the simulator's dark blue checker ground plane) and
                                    #            Pollen's own wall textures (from their empty vslam room), their light and sky
                                    # "white"  = plain white walls + light gray floor (WALL_RGBA / FLOOR_RGBA below)
WALL_RGBA = "0.96 0.96 0.96 1"      # white walls (not pure 1.0: that burns out in the camera picture)
FLOOR_RGBA = "0.76 0.76 0.76 1"     # light gray floor
WITH_FURNITURE = False              # sofa, table, bookshelf
WITH_DOOR_WINDOW = False            # the door and window panels on the walls
WITH_SMALL_THINGS = False           # sock, guitar, cup (the cup stands on the table, so it goes with it)
WITH_BALL = True                    # the loose ball he can kick


def box(name, x1, y1, x2, y2, z0, z1, material):
    """A solid box given by two corners in Anka's coordinates (it can be turned in the scene by to_mj)."""
    (ax, ay), (bx, by) = R.to_mj(x1, y1), R.to_mj(x2, y2)
    cx, cy = (ax + bx) / 2, (ay + by) / 2
    hx, hy, hz = abs(ax - bx) / 2, abs(ay - by) / 2, (z1 - z0) / 2
    return (f'    <geom name="{name}" type="box" size="{hx:.4f} {hy:.4f} {hz:.4f}" pos="{cx:.4f} {cy:.4f} {z0 + hz:.4f}" '
            f'material="{material}" contype="1" conaffinity="1"/>')


def build(rl_dir, robot_model="allcollisions"):
    robot = (Path(rl_dir).expanduser() / MODEL_SUBDIR / f"robot_{robot_model}.xml").resolve()
    W, H, T = R.ROOM_W, R.ROOM_H, 0.10          # room size and wall thickness
    wh = R.WALL_HEIGHT
    out = []
    out.append('<mujoco model="anka_room">')
    out.append(f'  <include file="{robot}"/>')
    if ROOM_STYLE == "pollen":
        # copied from Pollen's scene.xml (floor, light, sky) and vslam_room.xml (wall textures)
        out.append('''
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="160" elevation="-25" offwidth="640" offheight="480"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="floor" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <texture type="2d" name="vs_wall_n" builtin="checker" mark="random" random="0.45" rgb1="0.86 0.72 0.60" rgb2="0.72 0.58 0.46" markrgb="0.30 0.20 0.14" width="512" height="512"/>
    <texture type="2d" name="vs_wall_s" builtin="checker" mark="random" random="0.45" rgb1="0.62 0.74 0.86" rgb2="0.48 0.60 0.74" markrgb="0.16 0.24 0.36" width="512" height="512"/>
    <texture type="2d" name="vs_wall_e" builtin="checker" mark="random" random="0.45" rgb1="0.70 0.82 0.64" rgb2="0.56 0.70 0.50" markrgb="0.20 0.30 0.16" width="512" height="512"/>
    <texture type="2d" name="vs_wall_w" builtin="checker" mark="random" random="0.45" rgb1="0.86 0.78 0.60" rgb2="0.74 0.64 0.46" markrgb="0.32 0.26 0.14" width="512" height="512"/>
    <material name="wall_n" texture="vs_wall_n" texrepeat="5 2"/>
    <material name="wall_s" texture="vs_wall_s" texrepeat="5 2"/>
    <material name="wall_e" texture="vs_wall_e" texrepeat="4 2"/>
    <material name="wall_w" texture="vs_wall_w" texrepeat="4 2"/>
''')
        wall_mat = {"s": "wall_s", "n": "wall_n", "w": "wall_w", "e": "wall_e"}
        light = '    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>'
    else:
        out.append(f'''
  <visual>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.5 0.5 0.5" specular="0 0 0"/>
    <rgba haze="0.85 0.85 0.85 1"/>
    <global azimuth="160" elevation="-25" offwidth="640" offheight="480"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.85 0.85 0.85" rgb2="0.6 0.6 0.6" width="512" height="3072"/>
    <material name="floor" rgba="{FLOOR_RGBA}" reflectance="0.02"/>
    <material name="wall_white" rgba="{WALL_RGBA}" reflectance="0.0"/>
''')
        wall_mat = {k: "wall_white" for k in "snwe"}
        light = '    <light pos="0 0 3.5" dir="0 0 -1" directional="true" diffuse="0.3 0.3 0.3" specular="0 0 0"/>'
    out.append('''    <material name="sofa" rgba="0.35 0.45 0.68 1"/>
    <material name="table_wood" rgba="0.55 0.38 0.20 1"/>
    <material name="bookshelf_mat" rgba="0.28 0.18 0.10 1"/>
    <material name="door_mat" rgba="0.42 0.26 0.14 1"/>
    <material name="window_mat" rgba="0.65 0.85 1.0 1"/>
    <material name="ball_mat" rgba="0.95 0.45 0.10 1"/>
    <material name="sock_mat" rgba="0.85 0.15 0.20 1"/>
    <material name="guitar_mat" rgba="0.70 0.42 0.18 1"/>
    <material name="cup_mat" rgba="0.95 0.95 0.95 1"/>
  </asset>

  <worldbody>''')
    out.append(light)
    # floor: in Pollen's style the same infinite ground plane the plain simulator uses (what the walking policy
    # was trained on); in the white style a slab under the room
    if ROOM_STYLE == "pollen":
        out.append('    <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="floor" contype="1" conaffinity="1"/>')
    else:
        (fx1, fy1), (fx2, fy2) = R.to_mj(0, 0), R.to_mj(W, H)
        out.append(f'    <geom name="floor" type="box" size="{abs(fx1 - fx2) / 2:.3f} {abs(fy1 - fy2) / 2:.3f} 0.005" '
                   f'pos="{(fx1 + fx2) / 2:.3f} {(fy1 + fy2) / 2:.3f} -0.005" material="floor" contype="1" conaffinity="1"/>')
    # walls (their inner side is the room's edge)
    out.append(box("wall_south", -T, -T, W + T, 0.0, 0, wh, wall_mat["s"]))
    out.append(box("wall_north", -T, H, W + T, H + T, 0, wh, wall_mat["n"]))
    out.append(box("wall_west", -T, 0.0, 0.0, H, 0, wh, wall_mat["w"]))
    out.append(box("wall_east", W, 0.0, W + T, H, 0, wh, wall_mat["e"]))
    # the door and the window are panels on the wall: things he can look at and walk to
    if WITH_DOOR_WINDOW:
        dx, dy = R.WALL_THINGS["door"]
        out.append(box("door", dx - 0.02, dy - 0.45, dx, dy + 0.45, 0, 2.0, "door_mat"))
        wx, wy = R.WALL_THINGS["window"]
        out.append(box("window", wx - 0.6, wy - 0.02, wx + 0.6, wy, 0.9, 1.7, "window_mat"))
    # furniture
    for name, (x1, y1, x2, y2, h_cm) in (R.FURNITURE.items() if WITH_FURNITURE else []):
        out.append(box(name, x1, y1, x2, y2, 0, h_cm / 100.0, {"sofa": "sofa", "table": "table_wood",
                                                               "bookshelf": "bookshelf_mat"}[name]))
    # small things that stay where they are
    if WITH_SMALL_THINGS:
        sx, sy = R.THINGS["sock"]
        out.append(box("sock", sx - 0.05, sy - 0.02, sx + 0.05, sy + 0.02, 0, 0.012, "sock_mat"))
        gx, gy = R.THINGS["guitar"]
        out.append(box("guitar_body", gx - 0.15, gy - 0.07, gx + 0.15, gy + 0.07, 0, 0.30, "guitar_mat"))
        out.append(box("guitar_neck", gx - 0.02, gy - 0.02, gx + 0.02, gy + 0.02, 0.30, 0.55, "guitar_mat"))
        cx, cy = R.THINGS["cup"]
        mx, my = R.to_mj(cx, cy)
        out.append(f'    <geom name="cup" type="cylinder" size="0.04 0.045" pos="{mx:.4f} {my:.4f} {0.75 + 0.045:.4f}" '
                   f'material="cup_mat" contype="1" conaffinity="1"/>')
    # the ball is loose: it rolls when it is kicked
    bx, by = R.to_mj(*R.THINGS["ball"])
    if WITH_BALL:
        out.append(f'    <body name="ball" pos="{bx:.4f} {by:.4f} 0.036">')
        out.append('      <freejoint name="ball_free"/>')
        out.append('      <geom name="ball" type="sphere" size="0.035" mass="0.02" material="ball_mat" '
                   'contype="1" conaffinity="1" friction="0.8 0.005 0.0001"/>')
        out.append('    </body>')
    out.append('  </worldbody>')
    out.append('  <keyframe>')
    ball = f"{bx:.4f} {by:.4f} 0.036 1 0 0 0" if WITH_BALL else ""
    for name, (qpos, ctrl) in POSES.items():
        out.append(f'    <key name="{name}" qpos="{qpos}  {ball}" ctrl="{ctrl}"/>')
    out.append('  </keyframe>')
    out.append('</mujoco>')
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rl", default=os.path.expanduser("~/Pollen/microduck_rl"), help="where microduck_rl is")
    p.add_argument("--robot", default="allcollisions", choices=["allcollisions", "groundcontact"],
                   help="allcollisions: the whole body bumps into furniture (like Pollen's apartment). "
                        "groundcontact: only the feet touch things - the model the plain simulator uses; try it if he cannot walk")
    p.add_argument("--out", default=str(HERE / "scene" / "anka_room.xml"))
    a = p.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(a.rl, a.robot))
    # the robot's meshes: the model asks for them in a folder called assets next to the main scene file
    assets = out.parent / "assets"
    target = Path(a.rl).expanduser() / MODEL_SUBDIR / "assets"
    if assets.is_symlink() or assets.exists():
        assets.unlink() if assets.is_symlink() else None
    if not assets.exists():
        assets.symlink_to(target)
    print(f"wrote {out} (robot model: {a.robot})\nassets -> {target}")


if __name__ == "__main__":
    main()
