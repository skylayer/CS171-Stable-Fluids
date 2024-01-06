//
// Created by condo on 2024/1/1.
//

#include "fluidSimulator.h"

#include <iomanip>

#include "vtkm/cont/DataSet.h"
#include "vtkm/io/writer/VTKDataSetWriter.h"
#include "vtkm/cont/DataSetBuilderUniform.h"


typedef float                       Scalar;
typedef Eigen::Matrix<Scalar, 3, 1> Vector;

void WritePVDFile(const std::string& pvdFileName, int numTimeSteps, double timeStepLength) {
    std::ofstream pvdFile(pvdFileName);

    // 写入PVD文件的头部
    pvdFile << "<?xml version=\"1.0\"?>\n";
    pvdFile << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    pvdFile << "  <Collection>\n";

    for (int i = 0; i < numTimeSteps; ++i) {
        double time = i * timeStepLength;
        std::string fileName = "velocity_" + std::to_string(i) + ".vtk";

        // 写入每个时间步的条目
        pvdFile << "    <DataSet timestep=\"" << std::setprecision(5) << time
                << "\" group=\"\" part=\"0\" file=\"" << fileName << "\"/>\n";
    }

    // 写入PVD文件的尾部
    pvdFile << "  </Collection>\n";
    pvdFile << "</VTKFile>\n";

    // 关闭文件
    pvdFile.close();
}

int main() {
    FluidSimulator3D<Scalar> f(10, 0.1);

    // PVD file


    for (int i = 0; i < 100; i++) {
        f.add_force();
        f.step();

        auto velocity = f.get_velocity();

        // 获取Tensor的维度
        auto        dims  = velocity.dimensions();
        std::size_t dim_x = dims[0];
        std::size_t dim_y = dims[1];
        std::size_t dim_z = dims[2];

        // 从Eigen::Tensor提取数据到std::vector，为了和VTKm兼容
        std::vector<vtkm::Vec<Scalar, 3>> velocityData;
        velocityData.reserve(dim_x * dim_y * dim_z);

        for (int x = 0; x < dim_x; ++x) {
            for (int y = 0; y < dim_y; ++y) {
                for (int z = 0; z < dim_z; ++z) {
                    Vector v = velocity(x, y, z);
                    velocityData.emplace_back(v[0], v[1], v[2]);
                }
            }
        }

        // 将数据转换为VTKm的ArrayHandle
        vtkm::cont::ArrayHandle<vtkm::Vec<Scalar, 3>> velocityArrayHandle = vtkm::cont::make_ArrayHandle(velocityData);

        // 使用VTKm创建数据集
        vtkm::cont::DataSetBuilderUniform dataSetBuilder;
        vtkm::cont::DataSet               dataSet = vtkm::cont::DataSetBuilderUniform::Create(vtkm::Id3(dim_x, dim_y, dim_z));

        // 将速度数据添加到数据集中
        dataSet.AddField(vtkm::cont::Field("velocity", vtkm::cont::Field::Association::Points, velocityArrayHandle));

        auto file_name = "velocity_" + std::to_string(i) + ".vtk";

        // 写入VTK文件
        const vtkm::io::VTKDataSetWriter writer(file_name.c_str());
        writer.WriteDataSet(dataSet);
    }

    // 创建并写入PVD文件
    WritePVDFile("velocity.pvd", 100, 0.1);

    return 0;
}
