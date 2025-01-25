from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

def create_assignment_docx():
    # Create a new Document
    doc = Document()

    # Title (Centered and Bold)
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run('Assignment 11')
    run.bold = True
    run.font.size = Pt(14)  # Adjust font size if needed

    # Problem Statement (Bold)
    doc.add_heading('Problem Statement:', level=1)
    problem_statement = doc.add_paragraph('Apply MADALINE on the XOR gate with a topology of 2-2-1.')
    problem_statement.runs[0].bold = True

    # Description
    doc.add_heading('Description:', level=1)
    doc.add_paragraph(
        'MADALINE (Multiple ADAptive LINear Elements) is a neural network architecture that consists of multiple Adaline (Adaptive Linear Neuron) units. '
        'It is used for solving non-linear classification problems, such as the XOR gate, which cannot be solved by a single-layer perceptron.'
    )
    doc.add_paragraph(
        'In this assignment, MADALINE is applied to the XOR gate, which outputs 1 when the number of 1s in the input is odd and 0 otherwise. '
        'The network topology is 2-2-1, meaning there are 2 input neurons, 2 hidden neurons, and 1 output neuron.'
    )

    # Algorithm (Numbered Steps with Bold "Step n:")
    doc.add_heading('Algorithm:', level=1)
    algorithm_steps = [
        "Initialize the weights and biases of the MADALINE network randomly.",
        "For each input-output pair in the training data, compute the output of each Adaline unit in the hidden layer.",
        "Apply the activation function (e.g., step function) to the outputs of the hidden layer to determine the final output.",
        "Compare the predicted output with the actual output and calculate the error.",
        "Update the weights and biases of the Adaline units using the Least Mean Squares (LMS) learning rule.",
        "Repeat the process for a specified number of epochs or until the network converges.",
        "Evaluate the MADALINE network on the test data to measure its performance."
    ]

    for i, step in enumerate(algorithm_steps, start=1):
        step_paragraph = doc.add_paragraph()
        step_run = step_paragraph.add_run(f'Step {i}: ')
        step_run.bold = True  # Make "Step n:" bold
        step_paragraph.add_run(step)  # Add the rest of the step text

    # Source Code (Stub)
    doc.add_heading('Source Code:', level=1)
    source_code_stub = """
import numpy as np
import pandas as pd

from backend import create_madaline
from sklearn.model_selection import train_test_split

# Load the XOR gate dataset
X = pd.read_csv('3_XOR.csv')
y = X.pop('output')

# Prepare the training and testing data
X_train = X[4:].drop('input_1', axis=1)
X_test = X_train
y_train = y[4:]
y_test = y_train

# Create the MADALINE network with a 2-2-1 topology
MADALINE = create_madaline("2-2-1", 2025)

# Train the MADALINE network
MADALINE.fit(X_train, y_train, epochs=75, learning_rate=0.7869240351280281)

# Evaluate the MADALINE network on the test data
MADALINE.evaluate(X_test, y_test)
    """

    # Add the source code stub with monospace font and smaller font size
    code_paragraph = doc.add_paragraph()
    run = code_paragraph.add_run(source_code_stub)
    font = run.font
    font.name = 'Courier New'  # Monospace font
    font.size = Pt(10)  # Smaller font size for code

    # Output (Placeholder)
    doc.add_heading('Output:', level=1)
    doc.add_paragraph('[Placeholder for output image or graph]')

    # Report
    doc.add_heading('Report:', level=1)
    doc.add_heading('Performance of MADALINE:', level=2)
    doc.add_paragraph(
        '- MADALINE is effective for solving non-linear classification problems like the XOR gate.\n'
        '- It uses multiple Adaline units to create a decision boundary that can separate non-linearly separable data.\n'
        '- The algorithm has a time complexity of O(n^2) in the worst case, making it suitable for small to medium-sized datasets.'
    )

    doc.add_heading('Drawbacks of MADALINE:', level=2)
    doc.add_paragraph(
        '- Scalability: Due to its high time complexity, MADALINE is not suitable for very large datasets.\n'
        '- Sensitivity to Initialization: The performance of the algorithm can be highly dependent on the initial weights and biases.\n'
        '- Complexity: The architecture can become complex with more hidden layers and neurons, making it harder to train.'
    )

    doc.add_heading('Remedies:', level=2)
    doc.add_paragraph(
        '- For large datasets, consider using more scalable algorithms like multi-layer perceptrons (MLP) or convolutional neural networks (CNN).\n'
        '- Experiment with different initialization methods to improve the performance of the MADALINE network.\n'
        '- Use techniques like regularization or dropout to reduce overfitting and improve generalization.'
    )

    # Save the document
    doc.save('Assignment_11_MADALINE.docx')

# Run the function to generate the document
create_assignment_docx()