# MathSolver - Interactive Visualizer for Mathematical Problem Solving

This project provides an interactive tool for visualizing the step-by-step solution of mathematical problems. It incorporates features like graph representations, annotations, and attention visualization, making it an excellent educational tool for both students and educators.

## Features

1. **Step-by-Step Visualization**: Each step in the problem-solving process is broken down into manageable parts, complete with explanations and intermediate steps.
2. **Graph Representation**: Visualize problem structures using graph representations for enhanced understanding.
3. **Annotations**: Add detailed annotations and mathematical properties for every step.
4. **Interactive Controls**:
   - Navigate between steps with "Previous" and "Next" buttons.
   - Play/Pause the animation.
   - Adjust the speed of the animation with a slider.
5. **Attention Visualization**: Display attention weights (if provided) to highlight the focus areas during each step.

## Getting Started

### Prerequisites

Ensure you have the following Python packages installed:

- `torch`
- `torch_geometric`
- `networkx`
- `matplotlib`
- `numpy`
- `IPython`

You can install the required packages using pip:

```bash
pip install torch torch-geometric networkx matplotlib numpy ipython
```

### Running the Application

1. Clone or download the repository.
2. Navigate to the project directory.
3. Run the main script:

```bash
python <script_name>.py
```

Replace `<script_name>` with the filename of the script.

4. The interactive visualizer window will open, displaying the problem-solving process.

## Code Overview

### Key Components

1. **`SolvingStep` Class**:
   - Represents an individual step in the solution process.
   - Includes details like equations, operations, explanations, annotations, and mathematical properties.

2. **`AnnotationBox` Class**:
   - Handles annotations displayed in the visualization.
   - Supports multiple styles (e.g., info, warning, highlight).

3. **`InteractiveVisualizer` Class**:
   - Manages the visualization interface.
   - Includes setup for multiple axes (e.g., equation, graph, operation, attention) and control buttons.

4. **`create_detailed_solution_steps` Function**:
   - Parses the problem and generates a series of `SolvingStep` objects with detailed annotations and properties.

### Example Problem

The code demonstrates solving the equation `2x + 3 = 7` with the following steps:

1. Problem Analysis
2. Term Identification
3. Isolate Variable
4. Simplify Right Side
5. Solve for x

Each step includes annotations, intermediate steps, and relevant mathematical properties.

### Controls

- **Previous**: Navigate to the previous step.
- **Play/Pause**: Toggle the animation playback.
- **Next**: Navigate to the next step.
- **Speed Slider**: Adjust the animation speed.

## Customization

- **Adding New Problems**: Modify the `create_detailed_solution_steps` function to include custom problems and their solutions.
- **Graph Representation**: Provide a `networkx` graph object for visualizing custom problem structures.
- **Attention Weights**: Populate the `attention_weights` attribute of `SolvingStep` for visualizing attention matrices.

## License

This project is released under the MIT License. See `LICENSE` for details.

## Contribution

Contributions are welcome! Feel free to open issues or submit pull requests for bug fixes, features, or enhancements.

## Acknowledgments

- **PyTorch**: Used for tensor computations.
- **PyTorch Geometric**: For handling graph-based computations.
- **NetworkX**: For creating and manipulating graph structures.
- **Matplotlib**: For creating the interactive visualizations.
