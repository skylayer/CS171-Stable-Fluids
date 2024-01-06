#include "fluid.h"
#include "fmt/format.h"
#include <vtkSmartPointer.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkUniformGrid.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <sstream>

#define Fluid FluidCUDA

auto WritePVDFile(const std::string &pvdFileName, int numTimeSteps, double timeStepLength) -> void {
    std::ofstream pvdFile(pvdFileName);

    // 写入PVD文件的头部
    pvdFile << "<?xml version=\"1.0\"?>\n";
    pvdFile << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    pvdFile << "  <Collection>\n";

    for (int i = 0; i < numTimeSteps; ++i) {
        double      time     = i * timeStepLength;
        std::string fileName = "vectorField" + std::to_string(i) + ".vti";

        // 写入每个时间步的条目
        pvdFile << "    <DataSet timestep=\"" << std::setprecision(5) << time << "\" group=\"\" part=\"0\" file=\"" << fileName << "\"/>\n";
    }

    // 写入PVD文件的尾部
    pvdFile << "  </Collection>\n";
    pvdFile << "</VTKFile>\n";

    // 关闭文件
    pvdFile.close();
}


void add_force(Fluid &fluid, int i) {
    // 0.1 < x < 0.3, 0.4 < y < 0.6, 0.4 < z < 0.6
    for (int x = 0.2F * CELLS_X; x <= 0.3F * CELLS_X; x++) {
        for (int y = 0.45F * CELLS_Y; y <= 0.55F * CELLS_Y; y++) {
            for (int z = 0.45F * CELLS_Z; z <= 0.55F * CELLS_Z; z++) {
                fluid.add_U_x_force_at(z - 5, y, x, FORCE_SCALE * expf(-i / 80.0F));
            }
        }
    }

    // // 0.7 < x < 0.9, 0.4 < y < 0.6, 0.4 < z < 0.6
    for (int x = 0.7F * CELLS_X; x <= 0.8F * CELLS_X; x++) {
        for (int y = 0.45F * CELLS_Y; y <= 0.55F * CELLS_Y; y++) {
            for (int z = 0.45F * CELLS_Z; z <= 0.55F * CELLS_Z; z++) {
                fluid.add_U_x_force_at(z + 5, y, x, -FORCE_SCALE * expf(-i / 80.0F));
            }
        }
    }
}


int main() {

    Fluid fluid;
    fluid.init();

    for (int i = 1; i < CELLS_Z / 4; i++) {
        for (int j = 1; j < CELLS_Y / 4; j++) {
            for (int k = 1; k < CELLS_X / 4; k++) {
                fluid.add_source_at(i + CELLS_Y / 4, j + CELLS_Y / 4, k + CELLS_Y / 4, 0, 1.0F);
            }
        }
    }

#define CYCLES 1000

    // add_force(fluid, 0);

    // timeit
    time_t start, end;
    start = clock();

    for (int i = 0; i < CYCLES; i++) {
        add_force(fluid, i);

        fluid.step();
        fmt::print("Iteration {}\n", i);
        fmt::print("Average time per iteration: {}s\n", (double)(clock() - start) / CLOCKS_PER_SEC / (i + 1));
        continue;
        // if (i != 100) {
        //     continue;
        // }

        // Create and update your structured grid with the vector field data for the current time step
        // Create a new uniform grid
        vtkSmartPointer<vtkUniformGrid> uniformGrid = vtkSmartPointer<vtkUniformGrid>::New();

        // Set the dimensions of the grid
        uniformGrid->SetDimensions(CELLS_X, CELLS_Y, CELLS_Z);
        uniformGrid->SetOrigin(0, 0, 0);
        uniformGrid->SetSpacing(1.0F / CELLS_X, 1.0F / CELLS_Y, 1.0F / CELLS_Z);

        vtkSmartPointer<vtkFloatArray> scalars = vtkSmartPointer<vtkFloatArray>::New();
        scalars->SetNumberOfComponents(1);
        scalars->SetName("density");
        // for (int z = 0; z < CELLS_Z; z++) {
        //     for (int y = 0; y < CELLS_Y; y++) {
        //         for (int x = 0; x < CELLS_X; x++) {
        //             auto s = fluid.S_at(z, y, x, 0);
        //             scalars->InsertNextValue(s);
        //         }
        //     }
        // }

        vtkSmartPointer<vtkFloatArray> points = vtkSmartPointer<vtkFloatArray>::New();
        points->SetNumberOfComponents(3);
        points->SetName("velocity");
        for (int z = 0; z < CELLS_Z; z++) {
            for (int y = 0; y < CELLS_Y; y++) {
                for (int x = 0; x < CELLS_X; x++) {
                    auto sx = fluid.Ux_at(z, y, x);
                    auto sy = fluid.Uy_at(z, y, x);
                    auto sz = fluid.Uz_at(z, y, x);
                    points->InsertNextTuple3(sx, sy, sz);
                }
            }
        }
        uniformGrid->GetPointData()->SetVectors(points);
        // uniformGrid->GetPointData()->SetScalars(scalars);

        // Setup the writer and export the data for the current time step
        const vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
        writer->SetFileName(("vectorField" + fmt::to_string(i) + ".vti").c_str());
        writer->SetInputData(uniformGrid);
        writer->Write();
    }

    WritePVDFile("velocity.pvd", CYCLES, DT);

    return 0;
}
