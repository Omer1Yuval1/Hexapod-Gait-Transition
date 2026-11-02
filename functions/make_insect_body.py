import numpy as np

def make_insect_body(params=None, joints=None, body_parts=None, camera=True, disable_free_joint=False, camera_mode=None):
    
    xml = f'''
<mujoco model="locust_body">
    <body euler="{' '.join(map(str, params["body_euler"]))}" name="torso" pos="{' '.join(map(str, params["body_pos"]))}">
    '''
    
    if(not disable_free_joint):
        xml += '<joint damping="0" stiffness="0" armature="0.0" frictionloss="0.0" margin="0" limited="false" name="root" pos="0 0 0" type="free"/>'
    
    if(params["default"]["geom"]["density"] is None):
        xml += f'''
        <body name="head"><geom name="geom_head" mass="{body_parts["head"]["mass"]}" type="mesh" mesh="head" /></body>
        <body name="thorax"><geom name="geom_thorax" mass="{body_parts["thorax"]["mass"]}" type="mesh" mesh="thorax" /></body>
        <body name="abdomen"><geom name="geom_abdomen" mass="{body_parts["abdomen"]["mass"]}" type="mesh" mesh="abdomen" /></body>
    '''
    else:
        xml += f'''
        <body name="head"><geom name="geom_head" density="{params["default"]["geom"]["density"]}" type="mesh" mesh="head" /></body>
        <body name="thorax"><geom name="geom_thorax" density="{params["default"]["geom"]["density"]}" type="mesh" mesh="thorax" /></body>
        <body name="abdomen"><geom name="geom_abdomen" density="{params["default"]["geom"]["density"]}" type="mesh" mesh="abdomen" /></body>
    '''
    
    if(camera):
        pos = ' '.join(map(str, np.array([-15, 0, 1]) * params["total_length"] / 5))
        xml += f'\t<camera name="robot_view_0" mode="track" pos="{pos}" euler="90 -90 0" />\n'
        pos = ' '.join(map(str, np.array([0, -6, 12]) * params["total_length"] / 5))
        xml += f'\t\t<camera name="robot_view_1" mode="trackcom" pos="{pos}" euler="30 0 0"/>\n'
        # xml += '\t<camera name="track" mode="fixed" pos="-4 0 0.5" euler="90 -90 0"/>\n'
    
    # legs = ["L1", "R1", "L2", "R2", "L3", "R3"]
    leg_groups = ["forelegs", "middlelegs", "hindlegs"]
    sides = ["L", "R"]
    # leg_parts = ["coxa", "femur", "tibia", "tarsus"]
    
    for i in range(1,4): # For each leg pair.
        for j in range(2): # For each side (left/right).
            s = sides[j]
            
            leg_pair = leg_groups[i-1]
            
            xml+= f'''
        <body name="{s}{i}_coxa" pos="0 0 0">
            
            {get_geom(s+str(i), body_parts[leg_pair]["coxa"], "coxa", f"{s}{i}_0", params)}
            
            <body name="{s}{i}_femur">
                
                {get_geom(s+str(i), body_parts[leg_pair]["femur"], "femur", f"{s}{i}_1", params)}
                
                {get_joint(s+str(i), joints[s+str(i)]["J0"], "0")}
                {get_joint(s+str(i), joints[s+str(i)]["J1"], "1")}
                
                <body name="{s}{i}_tibia" pos="0 0 0">
                    '''
            
            if(i == 1 and params["animal"] == "mole_cricket"): # The foreleg tibia in the mole cricket gets friction.
                xml += get_geom(s+str(i), body_parts[leg_pair]["tibia"], "tibia", f"{s}{i}_2", params, set_friction=True)
            else:
                xml += get_geom(s+str(i), body_parts[leg_pair]["tibia"], "tibia", f"{s}{i}_2", params)
                
            xml+= f'''
                    {get_joint(s+str(i), joints[s+str(i)]["J2"], "2")}
                    
                <body name="{s}{i}_tarsus" pos="0 0 0">
                '''
            
            if(i == 1 and params["animal"] == "mole_cricket"): # The mole cricket's forelegs tarsus is represented using a sphere rather than a mesh.
                xml += f'''\t{get_geom(s+str(i), body_parts[leg_pair]["tarsus"], "tarsus", None, params, set_friction=False)}'''
            else:
                xml += f'''\t{get_geom(s+str(i), body_parts[leg_pair]["tarsus"], "tarsus", f"{s}{i}_345", params, set_friction=True)}'''
            
            xml += f'''
                    {get_joint(s+str(i), joints[s+str(i)]["J3"], "3", camera_mode=camera_mode)}
                    
                    </body>
                </body>
            </body>
        </body>
        '''
    xml += f'''
    </body>
</mujoco>
    '''
    
    return xml
    
def get_geom(leg, body_part, name, mesh, params, set_friction=False):
    
    mass = body_part["mass"]
    
    if(mesh is None):
        pos = ' '.join(f"{x:.3f}" for x in body_part[f"{leg}_pos"]) # ' '.join(map(str, body_part[f"{leg}_pos"]))
        size = ' '.join(map(str, body_part["size"]))
        # return f'<geom name="{leg}_{name}_geom" type="sphere" mass="{mass}" pos="{pos}" size="{size}" />'
    else:
        pos = None
        size = None
        
        if(set_friction):
            friction = ' '.join(map(str, body_part["friction"])) if hasattr(body_part["friction"], '__iter__') else str(body_part["friction"])
        
        '''
        if(not set_friction):
            # return f'<geom name="{leg}_{name}_geom" type="mesh" mesh="{mesh}" mass="{mass}" />' #  class="{leg}_{name}"
        else:
            # print(leg, name, set_friction)
            # print(leg, body_part)
            friction = ' '.join(map(str, body_part["friction"])) if hasattr(body_part["friction"], '__iter__') else str(body_part["friction"])
            # return f'<geom name="{leg}_{name}_geom" type="mesh" mesh="{mesh}" mass="{mass}" friction="{friction}" />' # class="{leg}_{name}"
        '''

    return (
        f'<geom name="{leg}_{name}_geom"'
        + (f' mass="{mass}"' if params["default"]["geom"]["density"] is None else f' density="{params["default"]["geom"]["density"]}"')
        + (f' type="mesh" mesh="{mesh}"' if mesh is not None else f' type="sphere" pos="{pos}" size="{size}"')
        + (f' friction="{friction}"' if set_friction else "")
        + " />"
    )
def get_joint(leg, joint, name, camera_mode=None):
    
    pos = ' '.join(f"{x:.3f}" for x in joint["pos"]) # ' '.join(map(str, joint["pos"]))
    axis = ' '.join(map(str, joint["axis"]))
    range = ' '.join(map(str, joint["range"]))
    ref = str(joint["ref"])
    springref = str(joint["springref"]) if "springref" in joint and joint["springref"] is not None else "0"
    
    frictionloss = str(joint["frictionloss"])
    armature = str(joint["armature"])
    stiffness = str(joint["stiffness"])
    damping = str(joint["damping"])
    # return f'<joint name="{leg}_{name}_joint" pos="{pos}" axis="{axis}" range="{range}" ref="{ref}" springref="{ref}" type="hinge" />'

    item = f'<joint name="{leg}_{name}_joint" pos="{pos}" axis="{axis}" range="{range}" ref="{ref}" springref="{springref}" type="hinge" frictionloss="{frictionloss}" armature="{armature}" stiffness="{stiffness}" damping="{damping}" />'

    if(not camera_mode is None and "tip_pos" in joint):
        pos = ' '.join(f"{x:.3f}" for x in joint["tip_pos"])
        item += f'\n\t\t\t\t\t<site name="{leg}_tip" pos="{pos}" size="0.01" />'
        

    return item