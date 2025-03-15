import os

def create_mesh(input_label_dir, output_mesh_dir, input_data_dir):
    for data_name in os.listdir(input_label_dir):
        input_data_path = os.path.join(input_data_dir, data_name) + "/"
        input_label_file = os.path.join(input_label_dir, data_name, "pre_label.txt")
        output_mesh_file = data_name + ".ply"
        print(input_data_path, '\n', output_mesh_dir, '\n', input_label_file, '\n', output_mesh_file)
        os.system("./build/DelaunayMeshing %s %s %s %s True" % (input_data_path, output_mesh_dir, input_label_file, output_mesh_file))

if __name__ == "__main__":
    input_data_path = "../example/processed_data/"
    output_mesh_path = "../example/exp/output_mesh/"
    input_label_path = "../example/exp/output_label/"
    os.makedirs(output_mesh_path, exist_ok=True)
    create_mesh(input_label_path, output_mesh_path, input_data_path)

