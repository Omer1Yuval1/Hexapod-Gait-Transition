def object_params(object_type):

    if object_type == 'tunnel_simple': # This creates to walls (box geoms) to the sides of the robot along the y-axis.
        P = [
            {"pos": "1.5 0 0.75", "size": "0.1 75 0.75"},
            {"pos": "-1.5 0 0.75", "size": "0.1 75 0.75"},
        ]
    elif object_type == 'tunnel': # This is the same as tunnel_simple, but with acclimation chambers.
        P = [
            {"pos": "0.7 -12 0", "size": "0.1 10 0.8"},
            {"pos": "-0.7 -12 0", "size": "0.1 10 0.8"},
            {"pos": "3 0.5 0", "size": "0.1 2.5 0.8"},
            {"pos": "-3 0.5 0", "size": "0.1 2.5 0.8"},
            {"pos": "1.85 -2 0", "size": "1.25 0.1 0.8"},
            {"pos": "-1.85 -2 0", "size": "1.25 0.1 0.8"},
            {"pos": "0 3 0", "size": "3.1 0.1 0.8"}
        ]
    elif object_type == 'wall_simple':
        P = [
            {"pos": "0 5 0", "size": "1 0.1 0.8"}
        ]
    elif object_type == 'jelly_box':
        P = [
            {"pos": "0 5 0", "size": "1 0.1 0.8"}
        ]

    return P