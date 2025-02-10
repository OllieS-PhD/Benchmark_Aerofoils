import h5py
import matplotlib.pyplot as plt

def dataLayout(data_path):
    # Open file and print the top level group names/keys
    # This data set contains 8996 unique airfoil shapes
    # The alpha+XX groups represents the airfoil angle of attack (4 and 12 degrees) and contain datasets for the coeffients of drag, lift, and moment
    # The shape group contains 4 datasets representing the different parameterizations of the airfoil geometries.
    with h5py.File(data_path, 'r') as hf:
        top_groups = [k for k in hf.keys()]
        print(top_groups)

    # Explore contents of the three top level groups

    # Each alpha+XX group contains 3 datasets and 1 group:
    #   C_d: dataset, (8996,) array representing the coeffients of drag for each airfoil geometry at the specified angle of attack
    #   C_l: dataset, (8996,) array representing the coeffients of lift for each airfoil geometry at the specified angle of attack
    #   C_m: dataset, (8996,) array representing the coeffients of moment for each airfoil geometry at the specified angle of attack
    #   flow_field: group containing 8996 subgroups associated with each airfoil and containing 7 datasets:
    #      x: (n_mesh,) array representing the x locations of the mesh points
    #      y: (n_mesh,) array representing the y locations of the mesh points
    #      rho: (n_mesh,) array representing the flow density at each mesh point
    #      rho_u: (n_mesh,) array representing the flow momentum in the x-direction at each mesh point
    #      rho_v: (n_mesh,) array representing the flow momentum in the y-direction at each mesh point
    #      e: (n_mesh,) array representing the total energy in the flow at each mesh point
    #      omega: (n_mesh,) array representing the flow vorticity at each mesh point
    #   Note that the number of mesh point n_mesh can be different for each airfoil
    # The shape group contains 4 members representing different paramerizations of the airfoil geometry:
    #   landmarks:  dataset (8996, 1001, 2) array representing the geometries of each airfoil via (x, y) landmarks.
    #   grassmann:  dataset (8996, 4) array representing the geometries of each airfoil via Grassmann representations.
    #   cst:        dataset (8996, 19) array representing the geometries of each airfoil via Class-Shape transformations.
    #   bezier:     dataset (8996, 15) array representing the goemetries of each airfoil via Bezier curves.

    with h5py.File(data_path, 'r') as hf:
        for name in top_groups:
            print('-------------------------------------------------')
            print('Group Name:', name)
            print('Group Info:', hf[name])
            print('')
            
            if 'alpha' in name:
                print('Aerodynamic quantities:')
                for key in hf[name].keys():
                    if 'flow' in key:
                        for i in range(3):
                            print('Flow Field data for airfoil {:04d}'.format(i))
                            for flow_key in hf[name][key]['{:04d}'.format(i)].keys():
                                print('  ', hf[name][key]['{:04d}'.format(i)][flow_key])    
                    else:
                        print(hf[name][key])
            else:
                print('Airfoil shape data:')
                for key in hf[name].keys():
                    print(hf[name][key])
        print('-------------------------------------------------')
        
    # Visualize the first 5 airfoil geometries using the landmarks dataset
    plt.figure(figsize=(10, 6))
    with h5py.File(data_path, 'r') as hf:
        landmarks = hf['shape']['landmarks'][()]
        af_shape_count = 0
        for landmark in landmarks[0:5]:
            plt.plot(landmark[:, 0], landmark[:, 1], label='airfoil {}'.format(af_shape_count))
            af_shape_count += 1
    plt.xlabel('x/c', fontsize=12)
    plt.ylabel('y/c', fontsize=12)
    plt.gca().set_aspect(1.)
    plt.legend(bbox_to_anchor=(1.25, 1.05), fontsize=12)
    plt.title('Sample Airfoil Geometries from Landmark Data')
    plt.show()
