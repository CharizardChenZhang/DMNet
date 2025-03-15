#include <iostream>
#include <string>
#include "create_delaunay_mesh.h"

int main(int argc, char** argv)
{
    std::string input_path = argv[1];
    std::string output_path = argv[2];
    std::string input_label_file_name = argv[3];
    std::string output_mesh_file_name = argv[4];
    std::string with_transform_str = argv[5];
    bool with_transform;
    if(with_transform_str == "0") {
        with_transform = false;
    } else {
        with_transform = true;
    }


    DelaunayMeshing delaunay_meshing_exe(input_path, output_path, input_label_file_name,
                                         output_mesh_file_name, with_transform);
    delaunay_meshing_exe.Run();

    std::cout << "Done!" << std::endl;
    return 0;
}
