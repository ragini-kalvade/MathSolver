import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import networkx as nx
import os

class EquationEncoder:
    def __init__(self, max_length=20):
        self.max_length = max_length
        self.char_to_idx = {
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
            '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'x': 10, '^': 11, '+': 12, '-': 13, '=': 14,
            ' ': 15
        }
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
    def encode_equation(self, equation):
        encoded = torch.zeros(self.max_length, dtype=torch.long)
        for i, char in enumerate(equation[:self.max_length]):
            if char in self.char_to_idx:
                encoded[i] = self.char_to_idx[char]
        return encoded
    
    def decode_equation(self, encoded):
        return ''.join(self.idx_to_char[idx.item()] for idx in encoded if idx.item() in self.idx_to_char)


class EquationType:
    LINEAR = 'linear'
    QUADRATIC = 'quadratic'


class EquationClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(16, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1])


class EquationSolver(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(16, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        return self.fc_layers(lstm_out[:, -1])


class MLMathSolver:
    def __init__(self):
        self.encoder = EquationEncoder()
        self.classifier = EquationClassifier(input_size=20, hidden_size=128)
        self.linear_solver = EquationSolver(input_size=20, hidden_size=128, output_size=1)
        self.quadratic_solver = EquationSolver(input_size=20, hidden_size=128, output_size=2)
        
    def load_model(self, model, model_path):
        """Load pre-trained models if available."""
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model file {model_path} not found.")
        return model
    
    def load_models(self):
        """Load all models."""
        self.classifier = self.load_model(self.classifier, 'classifier_model.pth')
        self.linear_solver = self.load_model(self.linear_solver, 'linear_solver_model.pth')
        self.quadratic_solver = self.load_model(self.quadratic_solver, 'quadratic_solver_model.pth')
    
    def parse_equation(self, equation_str):
        equation_str = equation_str.replace(" ", "").lower()
        is_quadratic = 'x^2' in equation_str or 'x²' in equation_str
        
        if is_quadratic:
            pattern = r'([-+]?\d*)?x\^2([-+]?\d*)?x?([-+]?\d+)?=0'
            match = re.match(pattern, equation_str)
            if match:
                a = float(match.group(1) or 1)
                b = float(match.group(2) or 0)
                c = float(match.group(3) or 0)
                return EquationType.QUADRATIC, (a, b, c)
        else:
            pattern = r'([-+]?\d*)?x([-+]?\d+)?=(\d+)'
            match = re.match(pattern, equation_str)
            if match:
                a = float(match.group(1) or 1)
                b = float(match.group(2) or 0)
                c = float(match.group(3))
                return EquationType.LINEAR, (a, b, c)
                
        raise ValueError("Invalid equation format")
    
    def solve_equation(self, equation_str):
    # Encode equation
        encoded_eq = self.encoder.encode_equation(equation_str)
        encoded_eq = encoded_eq.unsqueeze(0)  # Add batch dimension
        
        # Classify equation type
        with torch.no_grad():
            eq_type_logits = self.classifier(encoded_eq)
            eq_type_pred = torch.argmax(eq_type_logits, dim=1).item()
            
        # Solve based on type
        if eq_type_pred == 0:  # Linear
            with torch.no_grad():
                solution = self.linear_solver(encoded_eq)
                return [round(solution.item(), 4)]  # Ensure the output is rounded and formatted
        else:  # Quadratic
            with torch.no_grad():
                solutions = self.quadratic_solver(encoded_eq)
                return [round(sol, 4) for sol in solutions.tolist()[0]]  # Ensure the output is rounded and formatted

    
    def _solve_linear(self, encoded_eq):
        with torch.no_grad():
            solution = self.linear_solver(encoded_eq)
            return [solution.item()]
    
    def _solve_quadratic(self, encoded_eq):
        with torch.no_grad():
            solutions = self.quadratic_solver(encoded_eq)
            return solutions.tolist()[0]
    
    def train_models(self, training_data):
        """Training logic here (separated for future scalability)."""
        pass


class SolutionVisualizer:
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.fig = None
        self.solver = MLMathSolver()
        self.solver.load_models()
        
    def setup_figure(self):
        self.fig = plt.figure(figsize=self.figsize)
        gs = self.fig.add_gridspec(2, 2)
        
        # Create axes
        self.ax_equation = self.fig.add_subplot(gs[0, 0])
        self.ax_graph = self.fig.add_subplot(gs[0, 1])
        self.ax_solution = self.fig.add_subplot(gs[1, :])
        
        # Input box
        ax_input = plt.axes([0.1, 0.02, 0.3, 0.075])
        self.text_input = TextBox(ax_input, 'Equation:', initial='x + 5 = 10')
        self.text_input.on_submit(self.solve_equation)
        
        # Solve button
        ax_solve = plt.axes([0.45, 0.02, 0.1, 0.075])
        self.btn_solve = Button(ax_solve, 'Solve')
        self.btn_solve.on_clicked(lambda _: self.solve_equation(self.text_input.text))
        
        self.fig.suptitle('ML Math Equation Solver', fontsize=14)
    
    def solve_equation(self, equation_str):
        try:
            # Clear previous plots
            for ax in [self.ax_equation, self.ax_graph, self.ax_solution]:
                ax.clear()
                
            # Parse and solve equation
            eq_type, coefficients = self.solver.parse_equation(equation_str)
            solutions = self.solver.solve_equation(equation_str)
            
            # Display equation
            self.ax_equation.text(0.5, 0.5, f"Equation: {equation_str}",
                                fontsize=12, ha='center')
            self.ax_equation.axis('off')
            
            # Plot graph
            x = np.linspace(-10, 10, 200)
            if eq_type == EquationType.LINEAR:
                a, b, c = coefficients
                y = a * x + b
                self.ax_graph.plot(x, y, label=f"y = {a}x + {b}")
                self.ax_graph.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                self.ax_graph.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                self.ax_graph.grid(True, alpha=0.3)
                self.ax_graph.set_title('Linear Function')
            else:  # Quadratic
                a, b, c = coefficients
                y = a * x**2 + b * x + c
                self.ax_graph.plot(x, y, label=f"y = {a}x^2 + {b}x + {c}")
                self.ax_graph.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                self.ax_graph.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                self.ax_graph.grid(True, alpha=0.3)
                self.ax_graph.set_title('Quadratic Function')
            
            # Display solutions
            solution_text = "Solutions:\n" + "\n".join([f"x = {sol}" for sol in solutions])
            self.ax_solution.text(0.5, 0.5, solution_text,
                                fontsize=12, ha='center')
            self.ax_solution.axis('off')
            
            plt.draw()
        
        except ValueError as e:
            self.ax_solution.text(0.5, 0.5, f"Error: {str(e)}",
                                fontsize=12, ha='center', color='red')
            self.ax_solution.axis('off')
            plt.draw()
        
    def _display_results(self, equation_str, solutions, eq_type, coefficients):
        # Display equation
        self.ax_equation.text(0.5, 0.5, f"Equation: {equation_str}", fontsize=12, ha='center')
        self.ax_equation.axis('off')
        
        # Plot graph
        x = np.linspace(-10, 10, 200)
        if eq_type == EquationType.LINEAR:
            a, b, c = coefficients
            y = a * x + b
            self.ax_graph.plot(x, y)
            self.ax_graph.set_title('Linear Function')
        else:
            a, b, c = coefficients
            y = a * x**2 + b * x + c
            self.ax_graph.plot(x, y)
            self.ax_graph.set_title('Quadratic Function')
        
        # Display solutions
        solution_text = f"Solutions: {solutions}"
        self.ax_solution.text(0.5, 0.5, solution_text, fontsize=12, ha='center')
        self.ax_solution.axis('off')
        plt.draw()

def generate_training_data(num_samples=1000):
    linear_equations = []
    quadratic_equations = []
    
    for _ in range(num_samples):
        # Generate linear equations
        a = np.random.uniform(-10, 10)
        b = np.random.uniform(-10, 10)
        c = np.random.uniform(-10, 10)
        linear_eq = f"{a}x + {b} = {c}"
        solution = (c - b) / a
        linear_equations.append((linear_eq, solution))
        
        # Generate quadratic equations
        a = np.random.uniform(-10, 10)
        b = np.random.uniform(-10, 10)
        c = np.random.uniform(-10, 10)
        quadratic_eq = f"{a}x^2 + {b}x + {c} = 0"
        discriminant = b**2 - 4*a*c
        if discriminant >= 0:
            x1 = (-b + np.sqrt(discriminant)) / (2*a)
            x2 = (-b - np.sqrt(discriminant)) / (2*a)
            quadratic_equations.append((quadratic_eq, (x1, x2)))
            
    return linear_equations, quadratic_equations

def main():
    # Generate training data
    linear_data, quadratic_data = generate_training_data()
    
    # Create and train solver
    solver = MLMathSolver()
    solver.train_models({'linear': linear_data, 'quadratic': quadratic_data})
    
    # Create and display visualizer
    visualizer = SolutionVisualizer()
    visualizer.setup_figure()
    plt.show()

if __name__ == "__main__":
    main()