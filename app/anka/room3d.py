"""Anka's 3D picture of the room, saved as a web page you can turn around with the mouse.

Each little cube is a place where his depth AI and depth sensor saw something, in the colour his camera saw there.
In the simulation the real room can be shown too (thin lines), so you can see how close his picture is.
"""
import json

import numpy as np

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Anka's 3D room</title>
<style>
  :root { --bg: #f4f1ea; --ink: #2a2622; --muted: #6f675d; --panel: rgba(255,255,255,.85); }
  @media (prefers-color-scheme: dark) { :root { --bg: #1d1b19; --ink: #eee8dd; --muted: #a59d90; --panel: rgba(40,37,34,.85); } }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--ink); font: 14px/1.4 -apple-system, system-ui, sans-serif; overflow: hidden; }
  #info { position: absolute; left: 12px; top: 12px; max-width: min(360px, calc(100% - 24px)); background: var(--panel);
          padding: 10px 12px; border-radius: 10px; }
  #info h1 { font-size: 16px; margin: 0 0 4px; }
  #info p { margin: 4px 0; color: var(--muted); }
  label { display: block; margin-top: 4px; cursor: pointer; }
</style></head><body>
<div id="info">
  <h1>Anka's 3D picture of the room</h1>
  <p>__SUMMARY__</p>
  <p>Drag to turn · scroll to zoom · right-drag to move.</p>
  <label><input type="checkbox" id="real" checked> show the real room (thin lines)</label>
  <label><input type="checkbox" id="floor" checked> show the floor cubes</label>
  <label><input type="checkbox" id="path" checked> show his path</label>
</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const DATA = __DATA__;
const scene = new THREE.Scene();
const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
scene.background = new THREE.Color(dark ? 0x1d1b19 : 0xf4f1ea);
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.05, 100);
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(devicePixelRatio); renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
const cx = DATA.room[0] / 2, cz = DATA.room[1] / 2;
controls.target.set(cx, 0.4, cz);
camera.position.set(cx - 3.5, 4.2, cz - 4.5);
controls.update();
scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const sun = new THREE.DirectionalLight(0xffffff, 0.5); sun.position.set(3, 6, 2); scene.add(sun);
// his cubes (room x → x, height → y, room y → z)
function cubes(list) {
  const geo = new THREE.BoxGeometry(DATA.size * 0.92, DATA.size * 0.92, DATA.size * 0.92);
  const mat = new THREE.MeshLambertMaterial();
  const mesh = new THREE.InstancedMesh(geo, mat, Math.max(1, list.length));
  const m = new THREE.Matrix4(), c = new THREE.Color();
  list.forEach((p, i) => { m.setPosition(p[0], p[2], p[1]); mesh.setMatrixAt(i, m);
                           c.setRGB(p[3] / 255, p[4] / 255, p[5] / 255); mesh.setColorAt(i, c); });
  mesh.count = list.length;
  scene.add(mesh); return mesh;
}
const low = DATA.cubes.filter(p => p[2] < 0.05), high = DATA.cubes.filter(p => p[2] >= 0.05);
const floorMesh = cubes(low); cubes(high);
// the real room, as thin lines
const real = new THREE.Group();
const lineMat = new THREE.LineBasicMaterial({color: dark ? 0xd8cfc0 : 0x5a5248});
DATA.boxes.forEach(b => {
  const [x1, y1, x2, y2, z1, z2] = b;
  const g = new THREE.EdgesGeometry(new THREE.BoxGeometry(x2 - x1, z2 - z1, y2 - y1));
  const l = new THREE.LineSegments(g, lineMat); l.position.set((x1 + x2) / 2, (z1 + z2) / 2, (y1 + y2) / 2); real.add(l);
});
scene.add(real);
// his path
const pathPts = DATA.path.map(p => new THREE.Vector3(p[0], 0.01, p[1]));
const path = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pathPts), new THREE.LineBasicMaterial({color: 0xe8a33c}));
scene.add(path);
document.getElementById('real').onchange = e => real.visible = e.target.checked;
document.getElementById('floor').onchange = e => floorMesh.visible = e.target.checked;
document.getElementById('path').onchange = e => path.visible = e.target.checked;
addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
                                   renderer.setSize(innerWidth, innerHeight); });
(function loop() { requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();
</script></body></html>
"""


def real_boxes(body):
    """The real room as boxes (x1, y1, x2, y2, bottom, top) – only known in the simulation."""
    rw, rh = body.room_size()
    boxes = [(0.0, 0.0, rw, rh, 0.0, 2.4)]
    for (x1, y1, x2, y2, z0, z1, _, _) in body.solid_boxes():
        boxes.append((x1, y1, x2, y2, z0, z1))
    for (px, py) in body.people.values():
        boxes.append((px - 0.18, py - 0.18, px + 0.18, py + 0.18, 0.0, 1.75))
    return boxes


def how_right(cubes, boxes, near=0.075):
    """Share of his cubes that are within `near` m of a real surface (the floor, a wall, furniture, a person)."""
    if not len(cubes):
        return None
    x, y, z = cubes[:, 0], cubes[:, 1], cubes[:, 2]
    room = boxes[0]
    good = (z < near) & (x > -near) & (y > -near) & (x < room[2] + near) & (y < room[3] + near)     # the floor
    good |= ((np.abs(x) < near) | (np.abs(y) < near) | (np.abs(x - room[2]) < near) | (np.abs(y - room[3]) < near))  # walls
    for (x1, y1, x2, y2, z1, z2) in boxes[1:]:
        good |= ((x > x1 - near) & (x < x2 + near) & (y > y1 - near) & (y < y2 + near) & (z > z1 - near) & (z < z2 + near))
    return float(good.mean())


def save_3d(path, body, simulated=True):
    """Writes the web page. Returns a one-line summary."""
    cubes = body.cubes_3d()
    boxes = real_boxes(body) if simulated else []
    right = how_right(cubes, boxes) if simulated else None
    summary = (f"{len(cubes):,} cubes of 5 cm, built from what his depth AI and depth sensor saw while he walked "
               f"{body.walked:.1f} m." + (f" {right * 100:.0f} % of them are within 7.5 cm of something real." if right is not None else ""))
    rw, rh = body.room_size() if simulated else (5.0, 4.0)
    data = {"cubes": [[round(float(v), 3) for v in c[:3]] + [int(v) for v in c[3:]] for c in cubes],
            "boxes": [[round(float(v), 3) for v in b] for b in boxes], "size": 0.05, "room": [rw, rh],
            "path": [[round(float(a), 2), round(float(b), 2)] for a, b in getattr(body, "est_trail", None) or body.trail]}
    with open(path, "w", encoding="utf-8") as f:
        f.write(PAGE.replace("__SUMMARY__", summary).replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    return summary, right
