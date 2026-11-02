def make_insect_mesh(params):
    
    xml = f'''
<mujoco model="{params["animal"]}_mesh">
    '''
    
    mesh_rescale = ' '.join(map(str, params["mesh_rescale"]))

    body_parts = ["head", "thorax", "abdomen"] # , "antennas"
    for pair in ["1", "2", "3"]:
        for side in ["L", "R"]:
            for i in ["0", "1", "2", "345"]:
                if(not (params["animal"] == "mole_cricket" and pair == "1" and i == "345")): # Mole cricket has no mesh for the foreleg tarsi.
                    body_parts.append(f"{side}{pair}_{i}")
    
    for body_part in body_parts:
            xml+= f'''
    <mesh file="{params["version"]}/{body_part}.stl" scale="{mesh_rescale}"/>
        '''
    
    xml += '''
</mujoco>
'''

    return xml