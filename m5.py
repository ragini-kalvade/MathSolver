import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import networkx as nx

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

class DirectSolver:
    @staticmethod
    def solve_linear(a, b, c):
        """Solve linear equation ax + b = c"""
        return (c - b) / a
    
    @staticmethod
    def solve_quadratic(a, b, c):
        """Solve quadratic equation ax^2 + bx + c = 0"""
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None  # No real solutions
        elif discriminant == 0:
            x = -b / (2*a)
            return [x, x]
        else:
            x1 = (-b + np.sqrt(discriminant)) / (2*a)
            x2 = (-b - np.sqrt(discriminant)) / (2*a)
            return [x1, x2]

class MLMathSolver:
    def __init__(self):
        self.encoder = EquationEncoder()
        self.direct_solver = DirectSolver()
    
    def parse_equation(self, equation_str):
        """Parse equation and identify its type."""
        # Clean the equation string
        equation_str = equation_str.replace(" ", "").lower()
        
        # Check if equation is quadratic
        if 'x^2' in equation_str or 'x²' in equation_str:
            # Parse quadratic equation ax^2 + bx + c = 0
            equation_str = equation_str.replace('x²', 'x^2')
            pattern = r'([-+]?\d*)?x\^2([-+]?\d*)?x?([-+]?\d+)?=0'
            match = re.match(pattern, equation_str)
            if match:
                a = float(match.group(1) or 1)
                b = float(match.group(2) or 0)
                c = float(match.group(3) or 0)
                return EquationType.QUADRATIC, (a, b, c)
        else:
            # Parse linear equation ax + b = c
            pattern = r'([-+]?\d*)?x([-+]?\d+)?=(\d+)'
            match = re.match(pattern, equation_str)
            if match:
                a = float(match.group(1) or 1)
                b = float(match.group(2) or 0)
                c = float(match.group(3))
                return EquationType.LINEAR, (a, b, c)
                
        raise ValueError("Invalid equation format")
    
    def solve_equation(self, equation_str):
        """Solve the equation using direct algebraic methods."""
        eq_type, coefficients = self.parse_equation(equation_str)
        
        if eq_type == EquationType.LINEAR:
            a, b, c = coefficients
            solution = self.direct_solver.solve_linear(a, b, c)
            return [solution]
        else:  # Quadratic
            a, b, c = coefficients
            solutions = self.direct_solver.solve_quadratic(a, b, c)
            if solutions is None:
                return ["No real solutions"]
            return solutions

class SolutionVisualizer:
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.fig = None
        self.solver = MLMathSolver()
        
    def setup_figure(self):
        """Set up the interactive figure."""
        self.fig = plt.figure(figsize=self.figsize)
        gs = self.fig.add_gridspec(2, 2)
        
        # Create axes
        self.ax_equation = self.fig.add_subplot(gs[0, 0])
        self.ax_graph = self.fig.add_subplot(gs[0, 1])
        self.ax_solution = self.fig.add_subplot(gs[1, :])
        
        # Create input box
        ax_input = plt.axes([0.1, 0.02, 0.3, 0.075])
        self.text_input = TextBox(ax_input, 'Equation:', initial='x + 5 = 10')
        self.text_input.on_submit(self.solve_equation)
        
        # Create solve button
        ax_solve = plt.axes([0.45, 0.02, 0.1, 0.075])
        self.btn_solve = Button(ax_solve, 'Solve')
        self.btn_solve.on_clicked(lambda _: self.solve_equation(self.text_input.text))
        
        self.fig.suptitle('Math Equation Solver', fontsize=14)
        
    def solve_equation(self, equation_str):
        """Handle equation solving and visualization."""
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
                self.ax_graph.plot(x, y)
                self.ax_graph.axhline(y=c, color='r', linestyle='--', alpha=0.5)
                self.ax_graph.text(-9, c+0.5, f'y = {c}', color='r')
            else:  # Quadratic
                a, b, c = coefficients
                y = a * x**2 + b * x + c
                self.ax_graph.plot(x, y)
                self.ax_graph.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                self.ax_graph.text(-9, 0.5, 'y = 0', color='r')
            
            self.ax_graph.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            self.ax_graph.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            self.ax_graph.grid(True, alpha=0.3)
            self.ax_graph.set_title('Linear Function' if eq_type == EquationType.LINEAR else 'Quadratic Function')
            
            # Display solutions
            if isinstance(solutions[0], str):
                solution_text = solutions[0]  # "No real solutions" case
            else:
                solution_text = f"x = {', '.join(f'{sol:.2f}' for sol in solutions)}"
            
            self.ax_solution.text(0.5, 0.5, solution_text,
                                fontsize=12, ha='center')
            self.ax_solution.axis('off')
            
            plt.draw()
            
        except ValueError as e:
            self.ax_solution.text(0.5, 0.5, f"Error: {str(e)}",
                                fontsize=12, ha='center', color='red')
            self.ax_solution.axis('off')
            plt.draw()

def main():
    visualizer = SolutionVisualizer()
    visualizer.setup_figure()
    plt.show()

if __name__ == "__main__":
    main()