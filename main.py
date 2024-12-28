import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from IPython.display import HTML
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, Slider, TextBox

class SolvingStep:
    def __init__(self, equation, operation, explanation, graph=None):
        self.equation = equation
        self.operation = operation
        self.explanation = explanation
        self.graph = graph
        self.attention_weights = None
        self.highlighted_nodes = set()
        self.highlighted_edges = set()
        self.annotations = []  # List of detailed annotations
        self.intermediate_steps = []  # List of micro-steps within this step
        self.math_properties = []  # Mathematical properties used in this step
        
    def add_annotation(self, annotation):
        """Add a detailed annotation to the step."""
        self.annotations.append(annotation)
        
    def add_intermediate_step(self, step_description):
        """Add a micro-step description."""
        self.intermediate_steps.append(step_description)
        
    def add_math_property(self, property_name, explanation):
        """Add a mathematical property used in this step."""
        self.math_properties.append({
            'property': property_name,
            'explanation': explanation
        })

class AnnotationBox:
    def __init__(self, ax, text, position, style='info'):
        self.ax = ax
        self.text = text
        self.position = position
        self.style = style
        
        # Style configurations
        self.styles = {
            'info': {'facecolor': 'lightblue', 'alpha': 0.3},
            'warning': {'facecolor': 'lightyellow', 'alpha': 0.3},
            'highlight': {'facecolor': 'lightgreen', 'alpha': 0.3}
        }
        
    def draw(self):
        """Draw the annotation box."""
        style = self.styles.get(self.style, self.styles['info'])
        self.ax.text(
            self.position[0], self.position[1], self.text,
            bbox=dict(
                facecolor=style['facecolor'],
                alpha=style['alpha'],
                edgecolor='gray',
                boxstyle='round,pad=0.5'
            ),
            fontsize=10,
            wrap=True
        )

class InteractiveVisualizer:
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        self.current_step = 0
        self.steps = []
        self.fig = None
        self.axes = {}
        self.animation = None
        self.playing = False
        self.speed = 1.0
        
    def setup_figure(self):
        """Set up the figure with all necessary components."""
        self.fig = plt.figure(figsize=self.figsize)
        
        # Main content area
        gs = self.fig.add_gridspec(3, 2, height_ratios=[3, 3, 1])
        
        # Create axes for different components
        self.axes['equation'] = self.fig.add_subplot(gs[0, 0])
        self.axes['graph'] = self.fig.add_subplot(gs[0, 1])
        self.axes['operation'] = self.fig.add_subplot(gs[1, 0])
        self.axes['attention'] = self.fig.add_subplot(gs[1, 1])
        self.axes['controls'] = self.fig.add_subplot(gs[2, :])
        
        # Set up control buttons
        self.setup_controls()
        
    def setup_controls(self):
        """Set up interactive control buttons and sliders."""
        ax_controls = self.axes['controls']
        
        # Create button axes
        ax_prev = plt.axes([0.2, 0.05, 0.1, 0.075])
        ax_play = plt.axes([0.35, 0.05, 0.1, 0.075])
        ax_next = plt.axes([0.5, 0.05, 0.1, 0.075])
        ax_speed = plt.axes([0.7, 0.05, 0.2, 0.075])
        
        # Create buttons
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_play = Button(ax_play, 'Play/Pause')
        self.btn_next = Button(ax_next, 'Next')
        self.slider_speed = Slider(ax_speed, 'Speed', 0.1, 3.0, valinit=1.0)
        
        # Connect callbacks
        self.btn_prev.on_clicked(self.previous_step)
        self.btn_play.on_clicked(self.toggle_play)
        self.btn_next.on_clicked(self.next_step)
        self.slider_speed.on_changed(self.update_speed)
        
    def update_speed(self, val):
        """Update animation speed."""
        self.speed = val
        if self.animation:
            self.animation.event_source.interval = 2000 / val
            
    def toggle_play(self, event):
        """Toggle animation play/pause."""
        self.playing = not self.playing
        if self.playing and self.animation:
            self.animation.event_source.start()
        elif self.animation:
            self.animation.event_source.stop()
            
    def previous_step(self, event):
        """Go to previous step."""
        if self.current_step > 0:
            self.current_step -= 1
            self.update_display()
            
    def next_step(self, event):
        """Go to next step."""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self.update_display()
            
    def update_display(self):
        """Update the display for the current step."""
        step = self.steps[self.current_step]
        
        # Clear all axes
        for ax in self.axes.values():
            ax.clear()
            
        # Draw equation
        self._draw_equation(step)
        
        # Draw graph
        self._draw_graph(step)
        
        # Draw operation details
        self._draw_operation_details(step)
        
        # Draw attention weights
        self._draw_attention(step)
        
        # Add step annotations
        self._draw_annotations(step)
        
        plt.draw()
        
    def _draw_equation(self, step):
        """Draw equation with annotations."""
        ax = self.axes['equation']
        ax.text(0.5, 0.7, step.equation, fontsize=16, ha='center')
        
        # Add equations annotations
        for i, annotation in enumerate(step.annotations):
            AnnotationBox(ax, annotation, (0.1, 0.4 - i*0.15), 'info').draw()
            
        ax.set_title('Current Equation')
        ax.axis('off')
        
    def _draw_graph(self, step):
        """Draw graph with highlighted components."""
        ax = self.axes['graph']
        if step.graph:
            pos = nx.spring_layout(step.graph)
            nx.draw(step.graph, pos, ax=ax,
                   node_color='lightblue',
                   node_size=1500,
                   with_labels=True)
            
            # Highlight active nodes/edges
            if step.highlighted_nodes:
                nx.draw_networkx_nodes(step.graph, pos,
                                     nodelist=list(step.highlighted_nodes),
                                     node_color='yellow',
                                     node_size=1500,
                                     ax=ax)
                                     
        ax.set_title('Problem Graph')
        
    def _draw_operation_details(self, step):
        """Draw operation details with intermediate steps."""
        ax = self.axes['operation']
        
        # Main operation
        ax.text(0.5, 0.9, f"Operation: {step.operation}", fontsize=12, ha='center')
        
        # Intermediate steps
        for i, intermediate_step in enumerate(step.intermediate_steps):
            ax.text(0.1, 0.7 - i*0.1, f"• {intermediate_step}", fontsize=10)
            
        # Mathematical properties used
        if step.math_properties:
            ax.text(0.5, 0.4, "Mathematical Properties Used:", fontsize=11, ha='center')
            for i, prop in enumerate(step.math_properties):
                ax.text(0.1, 0.3 - i*0.1,
                       f"• {prop['property']}: {prop['explanation']}",
                       fontsize=10)
                       
        ax.set_title('Operation Details')
        ax.axis('off')
        
    def _draw_attention(self, step):
        """Draw attention weights with annotations."""
        ax = self.axes['attention']
        if step.attention_weights is not None:
            im = ax.imshow(step.attention_weights, cmap='YlOrRd')
            plt.colorbar(im, ax=ax)
            ax.set_title('Attention Weights')
        else:
            ax.text(0.5, 0.5, "No attention data", ha='center', va='center')
            ax.axis('off')
            
    def _draw_annotations(self, step):
        """Draw general annotations for the current step."""
        # Add step counter
        self.fig.suptitle(f'Step {self.current_step + 1} of {len(self.steps)}',
                         fontsize=14, y=0.98)

def create_detailed_solution_steps(problem_text):
    """Create solution steps with detailed annotations."""
    steps = []
    
    # Parse initial problem
    initial_step = SolvingStep(
        equation=problem_text,
        operation="Problem Analysis",
        explanation="Analyzing the initial equation",
        graph=nx.Graph()
    )
    initial_step.add_annotation("Original equation identified")
    initial_step.add_math_property(
        "Equation Properties",
        "An equation maintains equality when the same operation is performed on both sides"
    )
    steps.append(initial_step)
    
    # Example: Break down 2x + 3 = 7
    if "2x + 3 = 7" in problem_text:
        # Step 1: Identify terms
        step1 = SolvingStep(
            equation="2x + 3 = 7",
            operation="Term Identification",
            explanation="Identifying and categorizing terms",
            graph=nx.Graph()
        )
        step1.add_annotation("Left side: 2x (variable term) + 3 (constant term)")
        step1.add_annotation("Right side: 7 (constant term)")
        step1.add_intermediate_step("Identified variable term: 2x")
        step1.add_intermediate_step("Identified constant terms: 3 and 7")
        step1.add_math_property(
            "Like Terms",
            "Terms with the same variables raised to the same powers can be combined"
        )
        steps.append(step1)
        
        # Step 2: Isolate variable term
        step2 = SolvingStep(
            equation="2x = 7 - 3",
            operation="Isolate Variable",
            explanation="Moving constant terms to right side",
            graph=nx.Graph()
        )
        step2.add_annotation("Subtracted 3 from both sides")
        step2.add_intermediate_step("Move constant term (3) to right side")
        step2.add_intermediate_step("Combine like terms on right side")
        step2.add_math_property(
            "Addition Property of Equality",
            "Adding or subtracting the same quantity from both sides maintains equality"
        )
        steps.append(step2)
        
        # Step 3: Simplify right side
        step3 = SolvingStep(
            equation="2x = 4",
            operation="Simplify",
            explanation="Simplifying the right side",
            graph=nx.Graph()
        )
        step3.add_annotation("Simplified: 7 - 3 = 4")
        step3.add_intermediate_step("Evaluate right side: 7 - 3 = 4")
        steps.append(step3)
        
        # Step 4: Solve for x
        step4 = SolvingStep(
            equation="x = 2",
            operation="Solve for x",
            explanation="Dividing both sides by 2",
            graph=nx.Graph()
        )
        step4.add_annotation("Divided both sides by 2")
        step4.add_intermediate_step("Divide both sides by coefficient of x (2)")
        step4.add_math_property(
            "Division Property of Equality",
            "Dividing both sides by the same non-zero number maintains equality"
        )
        steps.append(step4)
    
    return steps

def main():
    # Create visualizer
    visualizer = InteractiveVisualizer()
    
    # Create solution steps
    problem = "2x + 3 = 7"
    steps = create_detailed_solution_steps(problem)
    visualizer.steps = steps
    
    # Setup and display
    visualizer.setup_figure()
    visualizer.update_display()
    plt.show()

if __name__ == "__main__":
    main()
