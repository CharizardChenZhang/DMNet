import os

def main(exe_file):
    Input_point_dir = "../example/sampled_pcd/"
    Input_surface_dir = "../example/gt_mesh/"        # If it is only for testing rather than training, GT mesh is not required, and you can place any mesh here.
    Output_gt_dir = "../example/processed_data/"
    for file in os.listdir(Input_point_dir):
        point_path = os.path.join(Input_point_dir, file)
        surface_path = os.path.join(Input_surface_dir, file)
        output_dir = os.path.join(Output_gt_dir, file.split(".")[0]) + "/"   # do not delete "/"
        os.makedirs(output_dir, exist_ok=True)
        os.system("%s %s %s %s 10 10000 0.0 3" % (exe_file, point_path, surface_path, output_dir))
                # sample_depth  point_num  noise_level  ref_num

if __name__ == "__main__":
    exe_file = "./build/igl_gt"
    main(exe_file)
