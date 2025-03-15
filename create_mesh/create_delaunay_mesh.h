#ifndef DELAUNAY_MESHING_H
#define DELAUNAY_MESHING_H

#include<fstream>  
#include<iostream>  
#include<string>
#include <unordered_set>
#include <unordered_map>
#include <iterator>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>


#include <open3d/Open3D.h>
typedef CGAL::Exact_predicates_inexact_constructions_kernel         K;
typedef CGAL::Triangulation_vertex_base_with_info_3<double, K>      Vb;
typedef CGAL::Delaunay_triangulation_cell_base_3<K>                 Cb;
typedef CGAL::Triangulation_data_structure_3<Vb, Cb>                Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds>                      Delaunay;

namespace std {



template<>
struct hash<Delaunay::Vertex_handle> {
    std::size_t operator()(const Delaunay::Vertex_handle& handle) const {
        return reinterpret_cast<std::size_t>(&*handle);
    }
};

template<>
struct hash<const Delaunay::Vertex_handle> {
    std::size_t operator()(const Delaunay::Vertex_handle& handle) const {
        return reinterpret_cast<std::size_t>(&*handle);
    }
};

template<>
struct hash<Delaunay::Cell_handle> {
    std::size_t operator()(const Delaunay::Cell_handle& handle) const {
        return reinterpret_cast<std::size_t>(&*handle);
    }
};

template<>
struct hash<const Delaunay::Cell_handle> {
    std::size_t operator()(const Delaunay::Cell_handle& handle) const {
        return reinterpret_cast<std::size_t>(&*handle);
    }
};


}


template <typename T>
T Percentile(const std::vector<T>& elems, const double p) {

  const int idx = static_cast<int>(std::round(p / 100 * (elems.size() - 1)));
  const size_t percentile_idx =
      std::max(0, std::min(static_cast<int>(elems.size() - 1), idx));

  std::vector<T> ordered_elems = elems;
  std::nth_element(ordered_elems.begin(),
                   ordered_elems.begin() + percentile_idx, ordered_elems.end());

  return ordered_elems.at(percentile_idx);
}

K::Point_3 EigenToCGAL(const Eigen::Vector3d& point) {
    return K::Point_3(point(0), point(1), point(2));
}
Eigen::Vector3d CGALToEigen(const K::Point_3& point) {
    return Eigen::Vector3d(point.x(), point.y(), point.z());
}



class DelaunayMeshing
{
public:
    DelaunayMeshing(std::string& input_path, std::string& output_path,
                    std::string& input_label_file_name,
                    std::string& output_mesh_file_name,
                    bool with_transform);

    Delaunay trianglation_;

    std::string input_path_;
    std::string input_label_file_name_;

    std::string output_path_;
    std::string output_mesh_file_name_;

    std::vector<int> cell_labels_;
    std::unordered_map<const Delaunay::Cell_handle, int> cell_labe_map_;
    Eigen::Vector4d transform_;
    bool with_transform_;

    void ReadData();

    void TransformMesh(open3d::geometry::TriangleMesh& input, Eigen::Vector4d transform);
    void CreateMesh();
    void Run();

};

DelaunayMeshing::DelaunayMeshing(std::string& input_path, std::string& output_path,
                std::string& input_label_file_name, std::string& output_mesh_file_name,
                                 bool with_transform) {
    input_path_ = input_path;
    output_path_ = output_path;
    input_label_file_name_ = input_label_file_name;
    output_mesh_file_name_ = output_mesh_file_name;
    with_transform_ = with_transform;
}

void DelaunayMeshing::ReadData() {

    {
        std::ifstream iFileT(input_path_ + "output_triangulation_binary",std::ios::in | std::ios::binary);
        if (iFileT.fail()) throw std::runtime_error("failed to open output_triangulation_binary");
        CGAL::set_binary_mode(iFileT);
        iFileT >> trianglation_;

        std::cout << "trianglation_.number_of_vertices() " << trianglation_.number_of_vertices() << std::endl;
        std::cout << "trianglation_.number_of_cells() " << trianglation_.number_of_cells() << std::endl;
        std::cout << "trianglation_.number_of_edges() " << trianglation_.number_of_edges() << std::endl;
        std::cout << "trianglation_.number_of_facets() " << trianglation_.number_of_facets() << std::endl;
    }


    {
        std::ifstream in_file(input_label_file_name_);
        if (in_file.fail()) throw std::runtime_error("failed to open " + input_label_file_name_);
        std::istream_iterator<std::string> begin_iter(in_file);
        std::istream_iterator<std::string> end_iter;
        int element_count = std::distance(begin_iter, end_iter);
        in_file.close();
        in_file.open(input_label_file_name_);
        if (in_file.fail()) throw std::runtime_error("failed to open " + input_label_file_name_);

        cell_labels_.resize(element_count);
        for(int i=0; i<element_count; i++) {
            in_file >> cell_labels_[i];
        }
        in_file.close();
        std::cout << "ref_point_label_.size() " << cell_labels_.size() << std::endl;
    }

    if(with_transform_) {


        {
            std::ifstream in_file(input_path_ + "transform.txt");
            if (in_file.fail()) throw std::runtime_error("failed to open transform.txt");
            in_file >> transform_(0);
            in_file >> transform_(1);
            in_file >> transform_(2);
            in_file >> transform_(3);

            in_file.close();
            std::cout << "ref_point_label_.size() " << cell_labels_.size() << std::endl;
        }
    }

}

void DelaunayMeshing::TransformMesh(open3d::geometry::TriangleMesh& input,
                                    Eigen::Vector4d transform) {
    Eigen::Vector3d cloud_origin;
    cloud_origin(0) = transform(0);
    cloud_origin(1) = transform(1);
    cloud_origin(2) = transform(2);
    double cloud_scale = transform(3);
    std::for_each(input.vertices_.begin(), input.vertices_.end(), [&](Eigen::Vector3d &point) { point = point * cloud_scale + cloud_origin; } );
}

void DelaunayMeshing::CreateMesh() {


    int cnt = 0;
    for (auto it = trianglation_.all_cells_begin();
         it != trianglation_.all_cells_end(); ++it) {
        cell_labe_map_[it] = cell_labels_[cnt];
        cnt++;
    }


    std::vector<Delaunay::Facet> surface_facets;
    std::vector<float> surface_facet_side_lengths;
    std::unordered_set<Delaunay::Vertex_handle> surface_vertices;

    for(auto facet_it=trianglation_.finite_facets_begin(); facet_it!=trianglation_.finite_facets_end(); ++facet_it) {
        int cell_label = cell_labe_map_.at(facet_it->first);
        int mirror_cell_label = cell_labe_map_.at(facet_it->first->neighbor(facet_it->second));
        if(cell_label == mirror_cell_label) {
            continue;
        }

        for(int i=0; i<3; i++) {
            const auto& vertex = facet_it->first->vertex(trianglation_.vertex_triple_index(facet_it->second, i));
            surface_vertices.insert(vertex);
        }


        const K::Triangle_3 triangle = trianglation_.triangle(*facet_it);
        const float max_squared_side_length =
                std::max( { (triangle[0] - triangle[1]).squared_length(),
                            (triangle[0] - triangle[2]).squared_length(),
                            (triangle[1] - triangle[2]).squared_length() } );
        surface_facet_side_lengths.push_back(std::sqrt(max_squared_side_length));

        if(cell_label == 2) {
            surface_facets.push_back(*facet_it);
        } else {
            surface_facets.push_back(trianglation_.mirror_facet(*facet_it));
        }

    }

    open3d::geometry::TriangleMesh result_mesh;
    std::unordered_map<const Delaunay::Vertex_handle, int> surface_vertex_indices;
    surface_vertex_indices.reserve(surface_vertices.size());
    result_mesh.vertices_.reserve(surface_vertices.size());

    for(const auto& vertex : surface_vertices) {
        result_mesh.vertices_.push_back(CGALToEigen(vertex->point()));
        surface_vertex_indices.emplace(vertex, surface_vertex_indices.size());
    }

    double max_side_length_factor = 5.0;
    double max_side_length_percentile = 95.0;

    const float max_facet_side_length = max_side_length_factor*
            Percentile(surface_facet_side_lengths, max_side_length_percentile);

   result_mesh.triangles_.reserve(surface_facets.size());

   for(int i=0; i<surface_facets.size(); i++) {
       if(surface_facet_side_lengths[i] > max_facet_side_length) {
           continue;
       }

       const auto& facet = surface_facets[i];
       result_mesh.triangles_.push_back( Eigen::Vector3i(
                                         surface_vertex_indices.at(facet.first->vertex(
                                                                       trianglation_.vertex_triple_index(facet.second, 0))),
                                         surface_vertex_indices.at(facet.first->vertex(
                                                                       trianglation_.vertex_triple_index(facet.second, 1))),
                                         surface_vertex_indices.at(facet.first->vertex(
                                                                       trianglation_.vertex_triple_index(facet.second, 2)))
                                         ) );
   }

   if(with_transform_) {
       TransformMesh(result_mesh, transform_);
   }
   open3d::io::WriteTriangleMesh(output_path_ + output_mesh_file_name_, result_mesh);
}


void DelaunayMeshing::Run() {
    ReadData();
    CreateMesh();
}

#endif // DELAUNAY_MESHING_H
