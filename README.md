# Real-Time Smoke Fluid Simulation Project

![example.png](images%2Fexample.png)

## Project Introduction
This project aims to implement a real-time smoke fluid simulation. All components have been configured to run on GPUs using CUDA programming, which can be examined in the `kernels` folder. The project uses advanced fluid dynamics algorithms and volumetric rendering techniques for realistic smoke visualization. It is divided into three main parts: Fluid Solver, Renderer, and an OpenGL-based interactive window (GL).

### Fluid Solver
The Fluid Solver contains the core algorithms for fluid motion, efficiently implemented in CUDA for GPU acceleration.

### Renderer
The Renderer applies volumetric rendering techniques, also optimized with CUDA for GPU, to achieve realistic visualization of smoke.

### OpenGL Window and Interaction - GL
The GL part creates a visualization window and handles user interactions, enabling users to view and interact with the smoke simulation.

## Installation and Execution

### Installing Dependencies
The project depends on several libraries listed in the `CMakeLists.txt` file. To install these libraries, follow these steps:

1. Download and install CMake if not already installed on your system.
2. Clone this repository to your local machine.
3. Open a command line tool and navigate to the project directory.
4. Run the `cmake` command to load the `CMakeLists.txt` file, which will automatically download and install the required libraries.

### Running the Program
After installing the libraries, follow these steps to run the project:

1. Locate the `gl` executable file in the project directory.
2. Run the `gl` file to start the program.

### User Interaction Guide
- The simulation starts automatically upon launching the program.
- **Pause/Resume Simulation**: Press the `P` key to pause the fluid motion. Press the `O` key to resume the simulation.
- **Camera Control**: Use `W`/`S`, `A`/`D`, `Space`, and `Ctrl` keys to move the camera for different views of the 3D simulation.

## Contributors
- Chen Junsheng
- Zhang Yuefeng

## License
ShanghaiTech CS171 Final Project
