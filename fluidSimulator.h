//
// Created by condo on 2024/1/1.
//

#ifndef RTSTABLEFLUIDS_FLUIDSIMULATOR_CUH
#define RTSTABLEFLUIDS_FLUIDSIMULATOR_CUH

#include "unsupported/Eigen/CXX11/Tensor"

template <typename Scalar>
class FluidSimulator3D {
public:
    typedef Eigen::Matrix<Scalar, 3, 1> Vector;
    typedef Eigen::Tensor<Scalar, 3>    Tensor3;

    // Constructor to initialize the simulation grid and time step
    FluidSimulator3D(int gridSize, Scalar timeStep, Scalar viscosity = 0.0001) :
        N(gridSize), dt(timeStep), dx(Scalar(1.0) / static_cast<Scalar>(gridSize)), viscosity(viscosity) {
        // Initialize or allocate tensors/matrices here if needed
        velocity.resize(N + 2, N + 2, N + 2);
        velocity.setConstant(Vector(0, 0, 0));
        density.resize(N + 2, N + 2, N + 2);
        density.setZero();
    }

private:
    int                      N;         // Size of the grid
    Scalar                   dt;        // Time step
    const Scalar             dx;        // Space step
    Scalar                   viscosity; // Viscosity coefficient
    Eigen::Tensor<Vector, 3> velocity;  // Velocity components
    Eigen::Tensor<Scalar, 3> density;   // Density

    template <typename T>
    void corner(Eigen::Tensor<T, 3> &field) {
        field(0, 0, 0)             = (field(1, 0, 0) + field(0, 1, 0) + field(0, 0, 1)) / 3;
        field(0, 0, N + 1)         = (field(1, 0, N + 1) + field(0, 1, N + 1) + field(0, 0, N)) / 3;
        field(0, N + 1, 0)         = (field(1, N + 1, 0) + field(0, N, 0) + field(0, N + 1, 1)) / 3;
        field(0, N + 1, N + 1)     = (field(1, N + 1, N + 1) + field(0, N, N + 1) + field(0, N + 1, N)) / 3;
        field(N + 1, 0, 0)         = (field(N, 0, 0) + field(N + 1, 1, 0) + field(N + 1, 0, 1)) / 3;
        field(N + 1, 0, N + 1)     = (field(N, 0, N + 1) + field(N + 1, 1, N + 1) + field(N + 1, 0, N)) / 3;
        field(N + 1, N + 1, 0)     = (field(N, N + 1, 0) + field(N + 1, N, 0) + field(N + 1, N + 1, 1)) / 3;
        field(N + 1, N + 1, N + 1) = (field(N, N + 1, N + 1) + field(N + 1, N, N + 1) + field(N + 1, N + 1, N)) / 3;
    }

    void boundary(Eigen::Tensor<Vector, 3> &field) {
        for (int x = 1; x <= N; x++) {
            for (int y = 1; y <= N; y++) {
                field(x, y, 0)     = field(x, y, 1).cwiseProduct(Vector(1, 1, -1));
                field(x, y, N + 1) = field(x, y, N).cwiseProduct(Vector(1, 1, -1));
            }
        }

        for (int x = 1; x <= N; x++) {
            for (int z = 1; z <= N; z++) {
                field(x, 0, z)     = field(x, 1, z).cwiseProduct(Vector(1, -1, 1));
                field(x, N + 1, z) = field(x, N, z).cwiseProduct(Vector(1, -1, 1));
            }
        }

        for (int y = 1; y <= N; y++) {
            for (int z = 1; z <= N; z++) {
                field(0, y, z)     = field(1, y, z).cwiseProduct(Vector(-1, 1, 1));
                field(N + 1, y, z) = field(N, y, z).cwiseProduct(Vector(-1, 1, 1));
            }
        }

        for (int x = 1; x <= N; x++) {
            field(x, 0, 0)         = (field(x, 0, 1) + field(x, 1, 0)) / 2;
            field(x, 0, N + 1)     = (field(x, 0, N) + field(x, 1, N + 1)) / 2;
            field(x, N + 1, 0)     = (field(x, N + 1, 1) + field(x, N, 0)) / 2;
            field(x, N + 1, N + 1) = (field(x, N + 1, N) + field(x, N, N + 1)) / 2;
        }

        for (int y = 1; y <= N; y++) {
            field(0, y, 0)         = (field(0, y, 1) + field(1, y, 0)) / 2;
            field(0, y, N + 1)     = (field(0, y, N) + field(1, y, N + 1)) / 2;
            field(N + 1, y, 0)     = (field(N + 1, y, 1) + field(N, y, 0)) / 2;
            field(N + 1, y, N + 1) = (field(N + 1, y, N) + field(N, y, N + 1)) / 2;
        }

        for (int z = 1; z <= N; z++) {
            field(0, 0, z)         = (field(0, 1, z) + field(1, 0, z)) / 2;
            field(0, N + 1, z)     = (field(0, N, z) + field(1, N + 1, z)) / 2;
            field(N + 1, 0, z)     = (field(N + 1, 1, z) + field(N, 0, z)) / 2;
            field(N + 1, N + 1, z) = (field(N + 1, N, z) + field(N, N + 1, z)) / 2;
        }

        // 8 corners
        corner(field);
    }

    void boundary(Eigen::Tensor<Scalar, 3> &field) {
        for (int x = 1; x <= N; x++) {
            for (int y = 1; y <= N; y++) {
                field(x, y, 0)     = field(x, y, 1);
                field(x, y, N + 1) = field(x, y, N);
            }
        }

        for (int x = 1; x <= N; x++) {
            for (int z = 1; z <= N; z++) {
                field(x, 0, z)     = field(x, 1, z);
                field(x, N + 1, z) = field(x, N, z);
            }
        }

        for (int y = 1; y <= N; y++) {
            for (int z = 1; z <= N; z++) {
                field(0, y, z)     = field(1, y, z);
                field(N + 1, y, z) = field(N, y, z);
            }
        }

        for (int x = 1; x <= N; x++) {
            field(x, 0, 0)         = (field(x, 0, 1) + field(x, 1, 0)) / 2;
            field(x, 0, N + 1)     = (field(x, 0, N) + field(x, 1, N + 1)) / 2;
            field(x, N + 1, 0)     = (field(x, N + 1, 1) + field(x, N, 0)) / 2;
            field(x, N + 1, N + 1) = (field(x, N + 1, N) + field(x, N, N + 1)) / 2;
        }

        for (int y = 1; y <= N; y++) {
            field(0, y, 0)         = (field(0, y, 1) + field(1, y, 0)) / 2;
            field(0, y, N + 1)     = (field(0, y, N) + field(1, y, N + 1)) / 2;
            field(N + 1, y, 0)     = (field(N + 1, y, 1) + field(N, y, 0)) / 2;
            field(N + 1, y, N + 1) = (field(N + 1, y, N) + field(N, y, N + 1)) / 2;
        }

        for (int z = 1; z <= N; z++) {
            field(0, 0, z)         = (field(0, 1, z) + field(1, 0, z)) / 2;
            field(0, N + 1, z)     = (field(0, N, z) + field(1, N + 1, z)) / 2;
            field(N + 1, 0, z)     = (field(N + 1, 1, z) + field(N, 0, z)) / 2;
            field(N + 1, N + 1, z) = (field(N + 1, N, z) + field(N, N + 1, z)) / 2;
        }

        // 8 corners
        corner(field);
    }

    template <typename T>
    auto diffuse(Eigen::Tensor<T, 3> const &field_prev) {
        Eigen::Tensor<T, 3> field = field_prev;

        Scalar a = dt * viscosity * N * N;

        for (int iter = 0; iter < 20; iter++) {
            for (int x = 1; x <= N; x++) {
                for (int y = 1; y <= N; y++) {
                    for (int z = 1; z <= N; z++) {
                        field(x, y, z) = (field_prev(x, y, z) + a * (field(x - 1, y, z) + field(x + 1, y, z) + field(x, y - 1, z) + field(x, y + 1, z) + field(
                            x,
                            y,
                            z - 1
                            ) + field(x, y, z + 1))) / (1 + 6 * a);
                    }
                }
            }
            boundary(field);
        }

        return field;
    }

    template <typename T>
    auto advect(Eigen::Tensor<T, 3> const &field_prev, Eigen::Tensor<Vector, 3> const &velo) {
        Eigen::Tensor<T, 3> field = field_prev;

        Scalar dt0 = dt * N;
        for (int x = 1; x <= N; x++) {
            for (int y = 1; y <= N; y++) {
                for (int z = 1; z <= N; z++) {
                    Vector pos(x, y, z);
                    pos            = pos - dt0 * velo(x, y, z);
                    pos            = pos.cwiseMax(Vector(0.5, 0.5, 0.5));
                    pos            = pos.cwiseMin(Vector(N + 0.5, N + 0.5, N + 0.5));
                    int    i       = static_cast<int>(pos(0));
                    int    j       = static_cast<int>(pos(1));
                    int    k       = static_cast<int>(pos(2));
                    Scalar s       = pos(0) - i;
                    Scalar t       = pos(1) - j;
                    Scalar u       = pos(2) - k;
                    field(x, y, z) = (1 - s) * ((1 - t) * ((1 - u) * field_prev(i, j, k) + u * field_prev(i, j, k + 1)) + t * ((1 - u) * field_prev(i, j + 1, k)
                        + u * field_prev(i, j + 1, k + 1))) + s * ((1 - t) * ((1 - u) * field_prev(i + 1, j, k) + u * field_prev(i + 1, j, k + 1)) + t * ((1 -
                        u) * field_prev(i + 1, j + 1, k) + u * field_prev(i + 1, j + 1, k + 1)));
                }
            }
        }
        boundary(field);

        return field;
    }

    auto divergence(Eigen::Tensor<Vector, 3> const &velo) {
        Eigen::Tensor<Scalar, 3> div(N + 2, N + 2, N + 2);
        for (int x = 1; x <= N; x++) {
            for (int y = 1; y <= N; y++) {
                for (int z = 1; z <= N; z++) {
                    div(x, y, z) = (velo(x + 1, y, z)(0) - velo(x - 1, y, z)(0) + velo(x, y + 1, z)(1) - velo(x, y - 1, z)(1) + velo(x, y, z + 1)(2) - velo(
                        x,
                        y,
                        z - 1
                        )(2)) / 2 / dx;
                }
            }
        }
        boundary(div);
        return div;
    }

    auto gradient(Eigen::Tensor<Scalar, 3> const &field) {
        Eigen::Tensor<Vector, 3> grad(N + 2, N + 2, N + 2);
        for (int x = 1; x <= N; x++) {
            for (int y = 1; y <= N; y++) {
                for (int z = 1; z <= N; z++) {
                    grad(x, y, z)(0) = (field(x + 1, y, z) - field(x - 1, y, z)) / (2 * dx);
                    grad(x, y, z)(1) = (field(x, y + 1, z) - field(x, y - 1, z)) / (2 * dx);
                    grad(x, y, z)(2) = (field(x, y, z + 1) - field(x, y, z - 1)) / (2 * dx);
                }
            }
        }
        boundary(grad);
        return grad;
    }

    auto project(Eigen::Tensor<Vector, 3> const &velo_prev) {
        static Eigen::Tensor<Scalar, 3> p(N + 2, N + 2, N + 2);
        // p.setZero();
        Eigen::Tensor<Vector, 3>   velo = velo_prev;
        Eigen::Tensor<Scalar, 3> &&div  = divergence(velo_prev);

        // Solve ∇²p = ∇·v with Gauss-Seidel method
        for (int iter = 0; iter < 20; iter++) {
            for (int x = 1; x <= N; x++) {
                for (int y = 1; y <= N; y++) {
                    for (int z = 1; z <= N; z++) {
                        p(x, y, z) = -div(x, y, z) + (p(x - 1, y, z) + p(x + 1, y, z) + p(x, y - 1, z) + p(x, y + 1, z) + p(x, y, z - 1) + p(x, y, z + 1)) / 6;
                    }
                }
            }
            boundary(p);
        }

        // Subtract the gradient of p
        velo = velo_prev - gradient(p);
        boundary(velo);

        return velo;
    }

public:
    void step() {
        auto w1 = diffuse(velocity);
        auto w2 = project(w1);
        auto w3 = advect(w2, w2);
        auto w4 = project(w3);

        velocity = w4;
    }

    void add_force() {
        // 0.1 < x < 0.3, 0.4 < y < 0.6, 0.4 < z < 0.6
        for (int x = N / 10; x <= 3 * N / 10; x++) {
            for (int y = 4 * N / 10; y <= 6 * N / 10; y++) {
                for (int z = 4 * N / 10; z <= 6 * N / 10; z++) {
                    velocity(x, y-1, z) += dt * Vector(2, 0, 0);
                }
            }
        }

        // 0.7 < x < 0.9, 0.4 < y < 0.6, 0.4 < z < 0.6
        for (int x = 7 * N / 10; x <= 9 * N / 10; x++) {
            for (int y = 4 * N / 10; y <= 6 * N / 10; y++) {
                for (int z = 4 * N / 10; z <= 6 * N / 10; z++) {
                    velocity(x, y+1, z) += dt * Vector(-2, 0, 0);
                }
            }
        }
    }

    auto get_velocity() {
        return velocity;
    }
};

typedef FluidSimulator3D<float> FluidSimulator3Df;

#endif //RTSTABLEFLUIDS_FLUIDSIMULATOR_CUH
