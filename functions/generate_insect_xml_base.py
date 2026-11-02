import numpy as np

def generate_insect_xml_base(P, mesh_path, body_path, joints, objects=None):
    
    xml = f'''
<mujoco model="insect">
    <compiler angle="degree" coordinate="local" inertiafromgeom="true" settotalmass="{P["total_mass"]}"/>
    <option gravity="{P["gravity"]}" integrator="{P["solver"]}" timestep="{P["dt"]}" noslip_iterations="{P["noslip_iterations"]}" />
    '''
    
    xml += f'''
    <default>
        <joint armature="{P["default"]["joint"]["armature"]}" frictionloss="{P["default"]["joint"]["frictionloss"]}" limited="true" margin="{P["default"]["joint"]["margin"]}" stiffness="{P["default"]["joint"]["stiffness"]}" damping="{P["default"]["joint"]["damping"]}" />
        <geom solref="{P["default"]["geom"]["solref"]}" solimp="{P["default"]["geom"]["solimp"]}" mass="{P["default"]["geom"]["mass"]}" condim="3" contype="{P["default"]["geom"]["contype"]}" conaffinity="{P["default"]["geom"]["conaffinity"]}" rgba="0.8 0.6 0.4 1" friction="{P["default"]["geom"]["friction"]}" margin="{P["default"]["geom"]["margin"]}" gap="{P["default"]["geom"]["gap"]}" />
    </default>
    
    <visual>
        <scale jointwidth="0.05" jointlength="1" />
    </visual>
    
    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="512" height="512"/>
        <material name="MatPlane" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    '''
    
    if(not mesh_path is None):
        xml += f'\t<include file="{mesh_path}" />'
    
    xml += f'''
    </asset>
    
    <worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        
        <camera name="worldview_0" mode="fixed" pos="{' '.join(map(str, np.array([0, -10, 25]) * P["total_length"] / 5))}" euler="45 0 0"/>
        <camera name="worldview_bwd" mode="fixed" pos="{' '.join(map(str, np.array([0, 10, 25]) * P["total_length"] / 5))}" euler="-45 0 180"/>
        <camera name="worldview_1" mode="fixed" pos="{' '.join(map(str, np.array([-10, 0, 0.5]) * P["total_length"] / 5))}" euler="90 -90 0"/>
        
        <geom contype="{P["default"]["floor"]["contype"]}" conaffinity="{P["default"]["floor"]["conaffinity"]}" condim="3" mass="{P["default"]["floor"]["mass"]}" friction="{' '.join(map(str, P["default"]["floor"]["friction"]))}" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="100 100 2" type="plane"/>
        
        <include file="{body_path}" />
    '''
    
    if(not objects is None):    
        xml += objects
    
    xml += '\t</worldbody>'
    
    xml += '\n\n\t<actuator>\n'
    
    for i in range(1,4):
        for j in ["L", "R"]:
            for k in range(4):
                # ctrlrange = ' '.join(map(str, joints[j+str(i)]["J"+str(k)]["ctrlrange"]))
                ctrlrange = ' '.join(map(str, joints[j+str(i)]["J"+str(k)].get("ctrlrange", P["default"]["joint"]["ctrlrange"])))
                xml += f'\t\t<motor name="Motor_{j}{i}_{k}" ctrllimited="true" ctrlrange="{ctrlrange}" joint="{j}{i}_{k}_joint" gear="{str(joints[j+str(i)]["J"+str(k)]["gear_ratio"])}"/>\n'
    
    xml += '    </actuator>'
    
    '''
    xml += '\n\n\t<contact>\n'
    
    leg_parts = ["coxa", "femur", "tibia", "tarsus"]
    for i in range(1,4):
        for j in ["L", "R"]:
            for k in range(len(leg_parts)-1):
                xml += f'\t\t<exclude name="{j}{i}_{leg_parts[k]}_{leg_parts[k+1]}" body1="{j}{i}_{leg_parts[k]}" body2="{j}{i}_{leg_parts[k+1]}"/>\n'
    
    xml += '\t</contact>'
    '''
    
    xml += '\n</mujoco>'
    
    return xml