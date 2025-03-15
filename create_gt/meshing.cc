#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Kernel/global_functions.h>
#include <fstream>
#include <open3d/Open3D.h>

#include <Eigen/Sparse>
#include <iostream>
#include <unordered_map>
#include <random>
using namespace std;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;


typedef CGAL::Triangulation_vertex_base_with_info_3<int, K> Vb;
typedef CGAL::Delaunay_triangulation_cell_base_3<K>                 Cb;
typedef CGAL::Triangulation_data_structure_3<Vb, Cb>                Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds, CGAL::Fast_location> Delaunay;

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


K::Point_3 EigenToCGAL(const Eigen::Vector3d& point) {
    return K::Point_3(point(0), point(1), point(2));
}
Eigen::Vector3d CGALToEigen(const K::Point_3& point) {
    return Eigen::Vector3d(point.x(), point.y(), point.z());
}


class AccumulatedPoint {
public:
    AccumulatedPoint()
        : num_of_points_(0),
          point_(0.0, 0.0, 0.0),
          normal_(0.0, 0.0, 0.0),
          color_(0.0, 0.0, 0.0) {}

public:
    void AddPoint(const open3d::geometry::PointCloud &cloud, int index) {
        point_ += cloud.points_[index];
        if (cloud.HasNormals()) {
            if (!std::isnan(cloud.normals_[index](0)) &&
                !std::isnan(cloud.normals_[index](1)) &&
                !std::isnan(cloud.normals_[index](2))) {
                normal_ += cloud.normals_[index];
            }
        }
        if (cloud.HasColors()) {
            color_ += cloud.colors_[index];
        }
        num_of_points_++;
    }

    Eigen::Vector3d GetAveragePoint() const {
        return point_ / double(num_of_points_);
    }

    Eigen::Vector3d GetAverageNormal() const { return normal_.normalized(); }

    Eigen::Vector3d GetAverageColor() const {
        return color_ / double(num_of_points_);
    }

public:
    int num_of_points_;
    Eigen::Vector3d point_;
    Eigen::Vector3d normal_;
    Eigen::Vector3d color_;
};

Eigen::Vector3d GetMinBound(const open3d::geometry::PointCloud & input_points)
{
    const std::vector<Eigen::Vector3d>& points = input_points.points_;
    if (points.size() == 0) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    auto itr_x = std::min_element(points.begin(), points.end(),
                                  [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(0) < b(0); });
    auto itr_y = std::min_element(points.begin(), points.end(),
                                  [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(1) < b(1); });
    auto itr_z = std::min_element(points.begin(), points.end(),
                                  [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(2) < b(2); });
    return Eigen::Vector3d((*itr_x)(0), (*itr_y)(1), (*itr_z)(2));
}

Eigen::Vector3d GetMaxBound(const open3d::geometry::PointCloud& input_points)
{

    const std::vector<Eigen::Vector3d>& points = input_points.points_;
    if (points.size() == 0) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    auto itr_x = std::max_element(points.begin(), points.end(),
                                  [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(0) < b(0); });
    auto itr_y = std::max_element(points.begin(), points.end(),
                                  [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(1) < b(1); });
    auto itr_z = std::max_element(points.begin(), points.end(),
                                  [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(2) < b(2); });
    return Eigen::Vector3d((*itr_x)(0), (*itr_y)(1), (*itr_z)(2));
}

void normlize_points(open3d::geometry::PointCloud &input, Eigen::Vector4d& transform, const double scale_factor) {
    Eigen::Vector3d min_bound = GetMinBound(input);   
    Eigen::Vector3d max_bound = GetMaxBound(input);
    double cloud_scale = std::max(std::max(max_bound(0)-min_bound(0), max_bound(1)-min_bound(1)), max_bound(2)-min_bound(2));
    cloud_scale *= scale_factor;
    Eigen::Vector3d cloud_origin;
    cloud_origin(0) = max_bound(0) - (cloud_scale + max_bound(0) - min_bound(0)) / 2.0;
    cloud_origin(1) = max_bound(1) - (cloud_scale + max_bound(1) - min_bound(1)) / 2.0;
    cloud_origin(2) = max_bound(2) - (cloud_scale + max_bound(2) - min_bound(2)) / 2.0;
    std::for_each(input.points_.begin(), input.points_.end(), [&](Eigen::Vector3d &point) { point = (point - cloud_origin) / cloud_scale; } );
    transform(0) = cloud_origin(0);
    transform(1) = cloud_origin(1);
    transform(2) = cloud_origin(2);
    transform(3) = cloud_scale;  
}


void normlize_points(std::vector<Eigen::Vector3d> &input, Eigen::Vector4d transform) {

    Eigen::Vector3d cloud_origin;
    cloud_origin(0) = transform(0);
    cloud_origin(1) = transform(1);
    cloud_origin(2) = transform(2);
    double cloud_scale = transform(3);
    std::for_each(input.begin(), input.end(), [&](Eigen::Vector3d &point) { point = (point - cloud_origin) / cloud_scale; } );
}

void add_noise(open3d::geometry::PointCloud &input,
               double noise_level) {

    Eigen::Vector3d voxel_min_bound = GetMinBound(input);
    Eigen::Vector3d voxel_max_bound = GetMaxBound(input);

//    std::default_random_engine generator;

    std::random_device rd;
    std::mt19937 generator(rd());
    std::vector<Eigen::Vector3d> noise(input.points_.size());
    // x
    {
        double interval = voxel_max_bound(0) - voxel_min_bound(0);
        std::normal_distribution<double> dis(0.0, 1.0);
        for(auto& e : noise) {
            e(0) = noise_level * interval * dis(generator);
        }
    }
    // y
    {
        double interval = voxel_max_bound(1) - voxel_min_bound(1);
        std::normal_distribution<double> dis(0.0, 1.0);
        for(auto& e : noise) {
            e(1) = noise_level * interval * dis(generator);
        }
    }
    // z
    {
        double interval = voxel_max_bound(2) - voxel_min_bound(2);
        std::normal_distribution<double> dis(0.0, 1.0);
        for(auto& e : noise) {
            e(2) = noise_level * interval * dis(generator);
        }
    }

    for(int i = 0; i < input.points_.size(); i++) {
        input.points_[i] += noise[i];
    }


}

void voxel_downsample(const open3d::geometry::PointCloud &input,       
                      open3d::geometry::PointCloud& output,
                      Eigen::Vector3d voxel_min_bound,
                      Eigen::Vector3d voxel_max_bound,
                      double voxel_size)
{
    if (voxel_size <= 0.0) {
        std::cout << "[VoxelDownSample] voxel_size <= 0." << std::endl;
    }
    Eigen::Vector3d voxel_size3 =
            Eigen::Vector3d(voxel_size, voxel_size, voxel_size);
    if (voxel_size * std::numeric_limits<int>::max() <
        (voxel_max_bound - voxel_min_bound).maxCoeff()) {
        std::cout << "[VoxelDownSample] voxel_size is too small." << std::endl;
    }
    std::unordered_map<Eigen::Vector3i, AccumulatedPoint,
                       open3d::utility::hash_eigen<Eigen::Vector3i>>
            voxelindex_to_accpoint;

    Eigen::Vector3d ref_coord;
    Eigen::Vector3i voxel_index;
    for (int i = 0; i < (int)input.points_.size(); i++) {
        ref_coord = (input.points_[i] - voxel_min_bound) / voxel_size; 
        voxel_index << int(floor(ref_coord(0))), int(floor(ref_coord(1))),
                int(floor(ref_coord(2)));
        voxelindex_to_accpoint[voxel_index].AddPoint(input, i);       
    }
    bool has_normals = input.HasNormals();
    bool has_colors = input.HasColors();
    for (auto accpoint : voxelindex_to_accpoint) {
        output.points_.push_back(accpoint.second.GetAveragePoint());  
        if (has_normals) {
            output.normals_.push_back(accpoint.second.GetAverageNormal());
        }
        if (has_colors) {
            output.colors_.push_back(accpoint.second.GetAverageColor());
        }
    }
    open3d::utility::LogDebug(
            "Pointcloud down sampled from {:d} points to {:d} points.",
            (int)input.points_.size(), (int)output.points_.size());
}


void uniform_sample(const open3d::geometry::PointCloud & input, open3d::geometry::PointCloud & output,
                    std::vector<int>& indices, const int fixed_num_points)      
{
    int points_num = input.points_.size();
    if(points_num <= fixed_num_points)    
    {
        output += input;
        return;
    }

    //std::shared_ptr<open3d::geometry::PointCloud> input_point = std::make_shared<open3d::geometry::PointCloud>();
    //*input_point = input;
    //const std::shared_ptr<open3d::geometry::PointCloud> input_points(input_point);
    //std::shared_ptr<open3d::geometry::TriangleMesh> tri_mesh = open3d::geometry::TriangleMesh::CreateSphere();
    //std::shared_ptr<open3d::geometry::PointCloud> sample_points = tri_mesh->SamplePointsPoissonDisk(fixed_num_points, 5, input_points);
    //output = *sample_points;

    std::unordered_set<int> index_set;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, points_num - 1);       
    while (index_set.size() < fixed_num_points)
    {
        int tmp_int = dis(gen);
        index_set.insert(tmp_int);
    }
    for (int e : index_set)
    {
        output.points_.push_back(input.points_[e]);
        if (input.HasNormals()) output.normals_.push_back(input.normals_[e]);
        if (input.HasColors()) output.colors_.push_back(input.colors_[e]);
        indices.push_back(e);
    }
}


void ReadData(const std::string& point_path, const std::string& surface_path,
              open3d::geometry::PointCloud& original_points,
              open3d::geometry::TriangleMesh& surface){
    std::cout<<("Reading from sampled points...")<<std::endl;
    open3d::io::ReadPointCloud(point_path, original_points);
    std::cout<<("Done!")<<std::endl;
    std::cout<<("Reading from surface...")<<std::endl;
    open3d::io::ReadTriangleMesh(surface_path, surface);
    std::cout<<("Done!")<<std::endl;
}

Delaunay ConstructDelaunay(const open3d::geometry::PointCloud& sampled_points){
    std::cout << "CreateVoxelSampledDelaunayTriangulation" << std::endl;
    std::vector<Delaunay::Point> delaunay_points(sampled_points.points_.size());
    for (size_t i = 0; i < sampled_points.points_.size(); ++i) {
        delaunay_points[i] =
                Delaunay::Point(sampled_points.points_[i][0], sampled_points.points_[i][1],
                                sampled_points.points_[i][2]);
    }
    std::vector<int> point_idxs(sampled_points.points_.size());
    std::iota(point_idxs.begin(), point_idxs.end(), 0);

    std::cout << "Create triangulartion" << std::endl;
    const auto triangulation = Delaunay( boost::make_zip_iterator(boost::make_tuple( delaunay_points.begin(), point_idxs.begin() )),
                                          boost::make_zip_iterator(boost::make_tuple( delaunay_points.end(), point_idxs.end() ) )  );
    return triangulation;

}

void compute_ref_points(Delaunay& t, open3d::geometry::PointCloud& ref_points, int ref_num) {

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    int cnt = 0;
    for(auto ch = t.all_cells_begin(); ch != t.all_cells_end(); ch++) {     

        std::vector<Eigen::Vector3d> vs;                         
        if(t.is_infinite(ch)) {                                 
            for(int i = 0; i < 4; i++) {
                auto vh = ch->vertex(i);                         
                if(t.is_infinite(vh)) {
                    continue;
                } else {
                    vs.push_back(Eigen::Vector3d(vh->point().x(), vh->point().y(), vh->point().z()));      
                }

            }
            for(int i = 0; i < 4; i++) {
                auto nei_ch = ch->neighbor(i);                  
                if(t.is_infinite(nei_ch)) {                      
                    continue;
                } else {                                         
                    auto mirror_vh = t.mirror_vertex(ch, i);
                    vs.push_back(Eigen::Vector3d(mirror_vh->point().x(), mirror_vh->point().y(), mirror_vh->point().z()));
                    break;
                }
            }

            Eigen::Vector3d ray = (vs[0] + vs[1] + vs[2]) / 3.0 - vs[3];   
            vs[3] =1.0 * ray + (vs[0] + vs[1] + vs[2]) / 3.0;               
        } else {
            for(int i = 0; i < 4; i++) {
                auto vh = ch->vertex(i);                        
                vs.push_back(Eigen::Vector3d(vh->point().x(), vh->point().y(), vh->point().z()));
            }
        }


        for(int i = 0; i < ref_num; i++) {
            Eigen::Vector3d tmp_ref;
            tmp_ref.setZero();                                  
            Eigen::Vector4d wei;
            for(int j = 0; j < 4; j++) {
                wei(j) = dis(generator);                          
            }
            wei = wei / wei.sum();                             
            for(int j = 0; j < 4; j++) {
                tmp_ref += vs[j] * wei[j];                       
            }
            ref_points.points_.push_back(tmp_ref);
        }

    }

}

void compute_normal_bary(open3d::geometry::TriangleMesh& surface, open3d::geometry::PointCloud& normal_bary) {

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    std::vector<Eigen::Vector3d> & tn = normal_bary.normals_;
    std::vector<Eigen::Vector3i> & fc = surface.triangles_;
    std::vector<Eigen::Vector3d> & vt = surface.vertices_;
    tn.clear();
    for(int i = 0; i < fc.size(); i++) {                 
        Eigen::Vector3d const& a = vt[fc[i](0)];         
        Eigen::Vector3d const& b = vt[fc[i](1)];
        Eigen::Vector3d const& c = vt[fc[i](2)];
        Eigen::Vector3d ab = b - a;                       
        Eigen::Vector3d bc = c - b;
        Eigen::Vector3d ca = a - c;

        Eigen::Vector3d fn = ab.cross(-ca);               
        fn.normalize();

        Eigen::Vector3d zero = Eigen::Vector3d(0, 0, 0);
        if (fn == zero) {
            continue;
        }

        tn.push_back(fn);
        Eigen::Vector3d bary = (a + b + c).array() / 3.0;
        normal_bary.points_.push_back(bary);
    }
}

void compute_signed_distance_ref(open3d::geometry::PointCloud& query_cloud,                  
                             open3d::geometry::PointCloud& dst_cloud,                      
                             std::vector<double>& sds) {
    open3d::geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(dst_cloud);

    std::vector<int> indices;
    std::vector<double> distance2;

     for(int i = 0; i < query_cloud.points_.size(); i++) {
        indices.clear();
        distance2.clear();
        Eigen::Vector3d query = query_cloud.points_[i];
        int num_of_searched = kdtree.SearchKNN(query, 1, indices, distance2);
        int nei_idx = indices.front();
        Eigen::Vector3d nei_n = dst_cloud.normals_[nei_idx];
        Eigen::Vector3d nei_v = dst_cloud.points_[nei_idx];
        Eigen::Vector3d dv = query - nei_v;
        double sd = dv.dot(nei_n);
        sds.push_back(sd);
    }
}

void color_points(open3d::geometry::PointCloud& cloud,
                  std::vector<double>& sds) {
    cloud.colors_.clear();

    for(int i = 0; i < cloud.points_.size(); i++) {
        double dis = sds[i];
        Eigen::Vector3d color;

        if(dis >= 0) {
            color(0) = 1.0;
            color(1) = 0.0;
            color(2) = 0.0;
        } else {
            color(0) = 0.0;
            color(1) = 1.0;
            color(2) = 0.0;
        }
        cloud.colors_.push_back(color);
    }
}

void label_points(std::vector<double>& sds, std::vector<int>& labels) {               

    for(int i = 0; i < sds.size(); i++) {
        double dis = sds[i];
        int label = (dis >= 0 ? 1 : 0) + 1;
        labels.push_back(label);
    }
}

void write_delaunay(Delaunay triangulation, std::string output_path) {
    {
        std::ofstream oFile_ocvi(output_path + "output_cell_vertex_idx.txt",std::ios::out);
        for (auto it = triangulation.all_cells_begin();
             it != triangulation.all_cells_end(); ++it)
        {
            for(int j=0; j<4; j++)
            {

                if (triangulation.is_infinite(it->vertex(j)))
                {
                    oFile_ocvi << -1 << " ";
                }
                else {
                    oFile_ocvi << it->vertex(j)->info() << " ";
                }
                if(j == 3) {
                    oFile_ocvi << std::endl;
                }
            }
        }
        oFile_ocvi.close();
    }


    {
        std::ofstream oFileT1(output_path + "output_triangulation_binary",std::ios::out | std::ios::binary);
        CGAL::set_binary_mode(oFileT1);
        oFileT1 << triangulation;
        oFileT1.close();

    }

    CGAL::Unique_hash_map<Delaunay::Cell_handle, int> cell_idx_map;
    int cell_count = 0;
    for (auto it = triangulation.all_cells_begin();
        it != triangulation.all_cells_end(); ++it)
    {
        cell_idx_map[it] = cell_count;
        cell_count++;
    }

    {
        std::ofstream oFileT_adj(output_path + "output_tetrahedron_adj",std::ios::out );
        for (auto it = triangulation.all_cells_begin();
            it != triangulation.all_cells_end(); ++it)
        {
            for(int i=0;i<4;i++)
            {
                auto neighbor_cell = it->neighbor(i);
                int neighbor_index = cell_idx_map[neighbor_cell];
                oFileT_adj<<neighbor_index<<" ";
            }
            oFileT_adj<<std::endl;
        }
        oFileT_adj.close();
    }

    {
        std::ofstream oFile_facet(output_path + "output_facet_vertex_idx.txt", std::ios::out);
        for (auto it = triangulation.finite_facets_begin(); it != triangulation.finite_facets_end(); ++it) {
            for (int i = 0; i < 4; i++) {
                if (i == it->second) {
                    continue;
                }
                oFile_facet << it->first->vertex(i)->info() << " ";
            }
            oFile_facet << std::endl;
        }
        oFile_facet.close();
    }

    {   
        std::ofstream oFile_ft(output_path + "output_facet_nei_cell.txt", std::ios::out);
        for (auto it = triangulation.finite_facets_begin(); it != triangulation.finite_facets_end(); ++it) {
            oFile_ft << cell_idx_map[it->first] << " " << cell_idx_map[it->first->neighbor(it->second)] << std::endl;
        }
        oFile_ft.close();
    }
}

double get_knn_from_points(open3d::geometry::PointCloud& cloud) {  
    open3d::geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(cloud);
    double distance_all;
    std::vector<int> indices;
    std::vector<double> distance;

    for (int i = 0; i < cloud.points_.size(); i++) {
        indices.clear();
        distance.clear();
        Eigen::Vector3d query = cloud.points_[i];
        int num_of_searched = kdtree.SearchKNN(query, 4, indices, distance);
        std::sort(distance.begin(), distance.end());
        double distance_average = (distance[1] + distance[2] + distance[3])/3.0;
        distance_all += distance_average;
    }
    int s = cloud.points_.size();
    double knn_average = distance_all * 1.0 / s;
    return knn_average;
}

void limit_cell_label(Delaunay& t, std::vector<int>& point_labels, int ref_num, vector<int>& cell_label) {
    int count = 0;
    vector<double> dis;                                      
    std::unordered_map<const Delaunay::Cell_handle, int> cell_index_map; 
    for (auto it = t.all_cells_begin(); it != t.all_cells_end(); it++) {
        cell_index_map[it] = count;
        double label_average = 0;
        for (int i = 0; i < ref_num; i++) {
            int j = count * ref_num + i;
            label_average += point_labels[j];
        }
        label_average = label_average * 1.0 / ref_num;
        if (label_average > 1.5) cell_label.push_back(2);
        else cell_label.push_back(1);

        count += 1;
    }
}

void label_cells(Delaunay& t, std::vector<double>& sds,
                 std::unordered_map<const Delaunay::Cell_handle, int>& cell_label_map) {

    int cnt = 0;
    for (auto it = t.all_cells_begin();
         it != t.all_cells_end(); ++it) {
        double ave = 0.0;
        if(t.is_infinite(it)) {
            std::vector<double> diss;
            for(int i = 0; i < 4; i++) {
                auto vit = it->vertex(i);
                if (t.is_infinite(vit)) {
                    continue;
                }
                int p_idx = vit->info();
                diss.push_back(sds[p_idx]);
            }
            for(int i = 0; i < diss.size(); i++) {
                ave += diss[i];
            }
            ave /= diss.size();
        } else {
            std::vector<double> diss;
            for(int i = 0; i < 4; i++) {
                auto vit = it->vertex(i);
                int p_idx = vit->info();
                diss.push_back(sds[p_idx]);
            }
            for(int i = 0; i < diss.size(); i++) {
                ave += diss[i];
            }
            ave /= diss.size();
        }
        cell_label_map[it] = ((ave > 0.0) ? 1 : 0) + 1;
        cnt++;
    }
}

void extract_surface(Delaunay& t, std::unordered_map<const Delaunay::Cell_handle, int>& cell_label_map,
                     open3d::geometry::TriangleMesh & result_mesh) {

    std::vector<Delaunay::Facet> surface_facets;
    std::unordered_set<Delaunay::Vertex_handle> surface_vertices;

    for(auto facet_it=t.finite_facets_begin(); facet_it!=t.finite_facets_end(); ++facet_it) {
        int cell_label = cell_label_map.at(facet_it->first);
        int mirror_cell_label = cell_label_map.at(facet_it->first->neighbor(facet_it->second));
        if(cell_label == mirror_cell_label) {
            continue;
        }

        for(int i=0; i<3; i++) {
            const auto& vertex = facet_it->first->vertex(t.vertex_triple_index(facet_it->second, i));
            surface_vertices.insert(vertex);
        }

        if(cell_label == 2) {
            surface_facets.push_back(*facet_it);
        } else {
            surface_facets.push_back(t.mirror_facet(*facet_it));
        }
    }


    std::unordered_map<const Delaunay::Vertex_handle, int> surface_vertex_indices;
    surface_vertex_indices.reserve(surface_vertices.size());

    for(const auto& vertex : surface_vertices) {
        result_mesh.vertices_.push_back(CGALToEigen(vertex->point()));
        surface_vertex_indices.emplace(vertex, surface_vertex_indices.size());
    }


    for(int i=0; i<surface_facets.size(); i++) {

        const auto& facet = surface_facets[i];
        result_mesh.triangles_.push_back( Eigen::Vector3i(
                                              surface_vertex_indices.at(facet.first->vertex(
                                                                            t.vertex_triple_index(facet.second, 0))),
                                              surface_vertex_indices.at(facet.first->vertex(
                                                                            t.vertex_triple_index(facet.second, 1))),
                                              surface_vertex_indices.at(facet.first->vertex(
                                                                            t.vertex_triple_index(facet.second, 2)))
                                              ) );
    }
}


int main(int argc, char* argv[]){
    std::string input_point_path = argv[1];
    std::string input_surface_path = argv[2];
    std::string output_path = argv[3];
    int sample_depth = std::stoi(argv[4]);
    int npoints = std::stoi(argv[5]);
    double noise_level = std::stod(argv[6]);
    int ref_num = std::stoi(argv[7]);

    std::cout << "ref_num " << ref_num << std::endl;

    double scale_factor = 1.1;
    double voxel_size = 1.0 / std::pow(2.0, sample_depth);  

    open3d::geometry::PointCloud original_points;
    open3d::geometry::TriangleMesh surface;
    ReadData(input_point_path, input_surface_path, original_points, surface);

    open3d::geometry::PointCloud noise_points;
    noise_points += original_points; // copy
    add_noise(noise_points, noise_level);

    open3d::geometry::PointCloud normalized_points;   
    normalized_points += noise_points;                                 
    Eigen::Vector4d transform;
    normlize_points(normalized_points, transform, scale_factor);         

    // open3d::geometry::PointCloud sampled_points;
    // Eigen::Vector3d voxel_min_bound(0.0, 0.0, 0.0);
    // Eigen::Vector3d voxel_max_bound(1.0, 1.0, 1.0);
    // voxel_downsample(normalized_points, sampled_points,
    //                  voxel_min_bound, voxel_max_bound,
    //                  voxel_size);


    open3d::geometry::PointCloud sampled_points1;
    std::vector<int> uniform_indices;
    uniform_sample(normalized_points, sampled_points1, uniform_indices, npoints);

    // double knn_average;
    // knn_average = get_knn_from_points(sampled_points1);

    Delaunay triangulation = ConstructDelaunay(sampled_points1);
    std::cout << "creat delaunay data done" << std::endl;

    open3d::geometry::PointCloud ref_points;
    compute_ref_points(triangulation, ref_points, ref_num);            

    open3d::geometry::PointCloud normal_bary;
    normlize_points(surface.vertices_, transform);
    compute_normal_bary(surface, normal_bary);                  
    std::cout << "compute normal_bary done" << std::endl;

    std::vector<double> signed_distances;
    compute_signed_distance_ref(ref_points, normal_bary, signed_distances);
    std::cout << "compute_signed_distance_ref done" << std::endl;
    std::vector<int> labels;
    label_points(signed_distances, labels);
    std::cout << "label_points done" << std::endl;  

    std::vector<int> cell_labels;
    limit_cell_label(triangulation, labels, ref_num, cell_labels);
    std::cout << "cell_label done" << std::endl;


    write_delaunay(triangulation, output_path);
    std::cout << "write_delaunay done" << std::endl;

    // open3d::io::WritePointCloudToPLY(output_path + "sur_points.ply", normal_bary);
    open3d::io::WritePointCloud(output_path + "noise_points.ply", noise_points);
    open3d::io::WritePointCloud(output_path + "sampled_points.ply", sampled_points1);
    // open3d::io::WriteTriangleMesh(output_path + "normalized_surface.ply", surface);
    {
        std::ofstream oFile_transform(output_path + "transform.txt", std::ios::out );
        oFile_transform<< transform <<std::endl;
        oFile_transform.close();
    }

    {
          std::ofstream oFile_gt_label(output_path + "ref_point_label.txt",std::ios::out );
          for (auto e : labels)
          {
              oFile_gt_label<< e <<std::endl;
          }
          oFile_gt_label.close();
    }


    // {
    //       std::ofstream oFile(output_path + "uniform_indices.txt",std::ios::out );
    //       for (auto e : uniform_indices)
    //       {
    //           oFile<< e <<std::endl;
    //       }
    //       oFile.close();
    // }

    {
        std::ofstream oFile_cell_label(output_path + "cell_label.txt", std::ios::out);
        for (auto e : cell_labels)
        {
            oFile_cell_label << e << std::endl;
        }
        oFile_cell_label.close();
    }

    std::cout << "done" << std::endl;
}
