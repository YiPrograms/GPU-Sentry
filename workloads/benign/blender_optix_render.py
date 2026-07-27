import math
import time

import bpy


def configure_cycles():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = False
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024

    prefs = bpy.context.preferences.addons["cycles"].preferences
    selected = "CUDA"
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            for device in prefs.devices:
                device.use = device.type != "CPU" and ("Tesla" in device.name or device.type in {"CUDA", "OPTIX"})
            if any(device.use for device in prefs.devices):
                selected = backend
                break
        except Exception:
            continue
    scene.cycles.device = "GPU"
    print(f"Cycles backend: {selected}")


def build_scene():
    bpy.ops.object.delete()
    bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, -1))
    floor = bpy.context.object
    mat = bpy.data.materials.new("mat_floor")
    mat.diffuse_color = (0.2, 0.22, 0.25, 1)
    floor.data.materials.append(mat)

    for i in range(40):
        x = (i % 8) - 3.5
        y = (i // 8) - 2.0
        z = 0.15 + 0.08 * (i % 5)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.35, location=(x, y, z))
        obj = bpy.context.object
        mat = bpy.data.materials.new(f"mat_{i}")
        mat.diffuse_color = (0.2 + (i % 3) * 0.25, 0.35 + (i % 5) * 0.1, 0.8 - (i % 4) * 0.12, 1)
        obj.data.materials.append(mat)

    bpy.ops.object.light_add(type="AREA", location=(0, -4, 7))
    light = bpy.context.object
    light.data.energy = 900
    light.data.size = 5

    bpy.ops.object.camera_add(location=(0, -8, 5), rotation=(math.radians(58), 0, 0))
    bpy.context.scene.camera = bpy.context.object


def main():
    configure_cycles()
    build_scene()
    end = time.time() + 60
    frame = 1
    while time.time() < end:
        bpy.context.scene.frame_set(frame)
        for obj in bpy.context.scene.objects:
            if obj.type == "MESH" and obj.name.startswith("Sphere"):
                obj.rotation_euler[2] += 0.15
        bpy.ops.render.render(write_still=False)
        frame += 1


main()
