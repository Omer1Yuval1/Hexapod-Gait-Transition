import numpy as np
# import os

from functions.make_insect_body import make_insect_body
from functions.make_insect_mesh import make_insect_mesh
from functions.generate_insect_xml_base import generate_insect_xml_base

def generate_mujoco_xml_insect(P=None, camera_mode=None):
    
    insect_body_xml = make_insect_body(P["model"]["params"], P["model"]["joints"], P["model"]["body_parts"], camera_mode=camera_mode)
    insect_mesh_xml = make_insect_mesh(P["model"]["params"])
    
    P["model"]["params"]["dt"] = P["dt"]
    P["model"]["params"]["solver"] = P["solver"]
    
    # Add static objects
    if(P["gaits"][0]["type"] == "jump" or P["multi_gait"]):
        # wall_ypos = -5.0
        # friction = ' '.join(map(str, P["model"]["params"]["default"]["floor"]["friction"]))
        # objects = f'\n\t\n\t\t\t<geom contype="{P["model"]["params"]["default"]["floor"]["contype"]}" conaffinity="{P["model"]["params"]["default"]["floor"]["conaffinity"]}" mass="{P["model"]["params"]["default"]["floor"]["mass"]}" friction="{friction}" name="bar_1_geom" type="box" size="5 0.25 1.0" rgba="0.8 0.2 0.2 1.0" pos="0.0 {wall_ypos} 1.0"/>\n\t\t'
        objects = ''
    elif(P["object_mode"]): # Tunnel with chambers.
        objects = ''
        for i, item in enumerate(P["objects"]):
            friction = ' '.join(map(str, P["model"]["params"]["default"]["floor"]["friction"]))
            objects += f'\n\t\n\t\t\t<geom contype="{P["model"]["params"]["default"]["floor"]["contype"]}" conaffinity="{P["model"]["params"]["default"]["floor"]["conaffinity"]}" mass="{P["model"]["params"]["default"]["floor"]["mass"]}" friction="{friction}" name="bar_{i}_geom" type="box" size="{item["size"]}" pos="{item["pos"]}" rgba="0.8 0.2 0.2 0.4" />\n\t\t'
        objects += '\n'
    else:
        objects = None
    
    with open(f'./envs/mujoco/assets/bodies/{P["animal"]}/body.xml', 'w') as file:
        file.write(insect_body_xml)
    
    with open(f'./envs/mujoco/assets/bodies/{P["animal"]}/mesh.xml', 'w') as file:
        file.write(insect_mesh_xml)
    
    base_xml = generate_insect_xml_base(P=P["model"]["params"], mesh_path=f'bodies/{P["animal"]}/mesh.xml', body_path=f'bodies/{P["animal"]}/body.xml', joints=P["model"]["joints"], objects=objects)
    
    with open(f'./envs/mujoco/assets/insect.xml', 'w') as file:
        file.write(base_xml)