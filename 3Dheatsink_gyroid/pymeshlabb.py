import pymeshlab


def example_apply_filter():
    # lines needed to run this specific example
    print('\n')


    # create a new MeshSet
    ms = pymeshlab.MeshSet()

    # load mesh
    ms.load_new_mesh("/workspace/MTO/3Dheatsink_gyroid/gyroid_surface_lattice.stl")

    # apply convex hull filter to the current selected mesh (last loaded)
    ms.meshing_remove_duplicate_faces()
    # alternatively:
    # ms.apply_filter('generate_convex_hull')


    # # save the current selected mesh
    # ms.save_current_mesh(output_path + "convex_hull.obj")

 

example_apply_filter()