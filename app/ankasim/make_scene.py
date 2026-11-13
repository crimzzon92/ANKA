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
import furniture as F

HERE = Path(__file__).resolve().parent
WALL_RGBA = "0.90 0.88 0.84 1"        # the walls: one plain, warm light colour (red green blue, 0-1)
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
    out.append('''
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="160" elevation="-25" offwidth="640" offheight="480"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="t_floor" builtin="checker" mark="cross" rgb1="0.76 0.60 0.36" rgb2="0.48 0.36 0.20" markrgb="0.18 0.12 0.06" width="1024" height="1024"/>
    <material name="floor" texture="t_floor" texrepeat="5 4" reflectance="0.03"/>
    <!-- the four walls: one plain colour (change WALL_RGBA at the top of make_scene.py) -->
    <material name="wall" rgba="{WALL_RGBA}" reflectance="0.0"/>
{FURNITURE_MATERIALS}
  </asset>

  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
''')
    # floor: one slab under the whole room, in MuJoCo's coordinates
    (fx1, fy1), (fx2, fy2) = R.to_mj(0, 0), R.to_mj(W, H)
    out.append(f'    <geom name="floor" type="box" size="{abs(fx1 - fx2) / 2:.3f} {abs(fy1 - fy2) / 2:.3f} 0.005" '
               f'pos="{(fx1 + fx2) / 2:.3f} {(fy1 + fy2) / 2:.3f} -0.005" material="floor" contype="1" conaffinity="1"/>')
    # walls (their inner side is the room's edge)
    out.append(box("wall_south", -T, -T, W + T, 0.0, 0, wh, "wall"))
    out.append(box("wall_north", -T, H, W + T, H + T, 0, wh, "wall"))
    out.append(box("wall_west", -T, 0.0, 0.0, H, 0, wh, "wall"))
    out.append(box("wall_east", W, 0.0, W + T, H, 0, wh, "wall"))
    # the room's things, as detailed models (furniture.py): sofa, table, bookshelf, door, window, guitar, cup, sock
    out.extend(F.everything())
    # the ball is loose: it rolls when it is kicked
    bx, by = R.to_mj(*R.THINGS["ball"])
    out.append(f'    <body name="ball" pos="{bx:.4f} {by:.4f} 0.036">')
    out.append('      <freejoint name="ball_free"/>')
    out.append('      <geom name="ball" type="sphere" size="0.035" mass="0.02" material="ball" '
               'contype="1" conaffinity="1" friction="0.8 0.005 0.0001"/>')
    out.append('    </body>')
    out.append('  </worldbody>')
    out.append('  <keyframe>')
    ball = f"{bx:.4f} {by:.4f} 0.036 1 0 0 0"
    for name, (qpos, ctrl) in POSES.items():
        out.append(f'    <key name="{name}" qpos="{qpos}  {ball}" ctrl="{ctrl}"/>')
    out.append('  </keyframe>')
    out.append('</mujoco>')
    text = "\n".join(out) + "\n"
    mats = "\n".join(f'    <material name="{n}" rgba="{r[0]} {r[1]} {r[2]} {r[3]}"/>' for n, r in F.MATERIALS.items())
    return text.replace("{WALL_RGBA}", WALL_RGBA).replace("{FURNITURE_MATERIALS}", mats)


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
