import bpy
import os
import math
import csv
#import pandas as pd

## Step 1
### 清除场景中多余的组件.
#bpy.data.objects.remove(bpy.data.objects['Camera'])
##bpy.data.objects.remove(bpy.data.objects['Cube'])
##bpy.data.objects.remove(bpy.data.objects['Light'])
#fbx_path = r"D:\Downloads\new\POLYWINK_AIX_52_SAMPLE\POLYWINK_AIX_SAMPLE\POLYWINK_AIX_SAMPLE.fbx"
#bpy.ops.import_scene.fbx(filepath=fbx_path)

# Step 2
output_dir = r'D:\Downloads\new\POLYWINK_AIX_52_SAMPLE\POLYWINK_AIX_SAMPLE\result_'

obj = bpy.context.active_object
#shape_key = obj.data.shape_keys.key_blocks["eyeBlinkLeft"]
#keys = obj.data.shape_keys.key_blocks.keys()
#shape_key_index = keys.index(shape_key.name)
#print(shape_key, keys, shape_key_index)
#print(bpy.data.objects["POLYWINK_Bella"].data)

obj.data.shape_keys.key_blocks["eyeBlinkLeft"].value=1.0
print(obj.data.shape_keys.key_blocks["eyeBlinkLeft"].value)
#print(obj.data.shape_keys.key_blocks["eyeBlinkRight"])


bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
bpy.context.scene.display.shading.light = 'MATCAP'
bpy.context.scene.display.render_aa = 'FXAA'
bpy.context.scene.render.resolution_x=int(1024)
bpy.context.scene.render.resolution_y=int(768)
bpy.context.scene.render.fps = 25
bpy.context.scene.render.image_settings.file_format='PNG'
bpy.context.scene.render.filepath=os.path.join(output_dir, '{}.png'.format('Test'))

## Camera
bpy.ops.object.camera_add(enter_editmode=False, location=[0, -1, 1.1], rotation=[math.radians(90), 0, 0])
cam = bpy.data.objects['Camera']
cam.scale = [2, 2, 2]
bpy.context.scene.camera = cam # add cam so it's rendered

bpy.ops.render.render(write_still=True)


#blendshape_example_path = r"D:\Code\AIWIN\TEST\a10.csv"
#result = []
#with open(blendshape_example_path, "r") as f:
#    reader = csv.reader(f)
#    for row in reader:
#        result.append(row)
#head = [str.lower(item[0]) + item[1:] for item in result[0]]
#print(head)
#result = result[1:]

#print(head, len(result))


#for i in range(len(result)):
#    for j in range(52):
#        obj.data.shape_keys.key_blocks[head[j]].value = eval(result[i][j])
#    bpy.context.scene.render.filepath=os.path.join(output_dir, '{}.png'.format(i))
#    bpy.ops.render.render(write_still=True)

# Step 3
# Blender将图片序列+声音合成为视频
# https://jingyan.baidu.com/article/375c8e198d4a1464f2a2299c.html
