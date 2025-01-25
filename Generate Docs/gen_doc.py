from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

def create_assignment_docx():
    # Create a new Document
    doc = Document()

    # Title (Centered and Bold)
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run('Assignment 9')
    run.bold = True
    run.font.size = Pt(14)  # Adjust font size if needed

    # Problem Statement (Bold)
    doc.add_heading('Problem Statement:', level=1)
    problem_statement = doc.add_paragraph('Apply Fuzzy C-Means clustering on the Boston Housing dataset.')
    problem_statement.runs[0].bold = True

    # Description
    doc.add_heading('Description:', level=1)
    doc.add_paragraph(
        'Fuzzy C-Means (FCM) is a clustering algorithm that allows data points to belong to multiple clusters with varying degrees of membership. '
        'Unlike traditional clustering methods like K-Means, FCM assigns a membership value to each data point for each cluster, which represents the degree of belongingness.'
    )
    doc.add_paragraph(
        'The Boston Housing dataset is used in this assignment. It contains information about housing prices in the Boston area, with features such as crime rate, number of rooms, and accessibility to highways. '
        'The dataset is standardized before applying the FCM algorithm to ensure that all features contribute equally to the clustering process.'
    )

    # Algorithm (Numbered Steps with Bold "Step n:")
    doc.add_heading('Algorithm:', level=1)
    algorithm_steps = [
        "Standardize the dataset to ensure that all features have the same scale.",
        "Define the number of clusters and the fuzziness parameter (m).",
        "Initialize the membership matrix randomly or using a predefined method.",
        "Calculate the cluster centroids based on the current membership values.",
        "Update the membership values for each data point based on the distance to the cluster centroids.",
        "Repeat the centroid calculation and membership update steps until convergence or a maximum number of iterations is reached.",
        "Assign each data point to the cluster with the highest membership value.",
        "Visualize the clusters and evaluate the clustering performance using metrics like the Fuzzy Partition Coefficient (FPC)."
    ]

    for i, step in enumerate(algorithm_steps, start=1):
        step_paragraph = doc.add_paragraph()
        step_run = step_paragraph.add_run(f'Step {i}: ')
        step_run.bold = True  # Make "Step n:" bold
        step_paragraph.add_run(step)  # Add the rest of the step text

    # Source Code (Stub)
    doc.add_heading('Source Code:', level=1)
    source_code_stub = """
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the Boston Housing dataset
df = pd.read_csv("F:\\College Crap\\AI\\CU-MSC-AI-SEM3\\Dataset\\HousingData.csv")

# Standardize the data (important for clustering)
scaler = StandardScaler()
data_std = scaler.fit_transform(df)

# Define the number of clusters
n_clusters = 3

# Perform Fuzzy C-Means clustering
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data_std.T, n_clusters, 2, error=0.005, maxiter=1000, init=None
)

# Predict cluster membership for each data point
cluster_membership = np.argmax(u, axis=0)

# Add the cluster membership to the original dataframe
df['Cluster'] = cluster_membership

# Print the first few rows of the dataframe with cluster assignments
print(df.head())

# Plot the clusters (for 2D visualization, using the first two features)
plt.scatter(data_std[:, 0], data_std[:, 1], c=cluster_membership, cmap='viridis')
plt.title('Fuzzy C-Means Clustering on Boston Housing Dataset')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.show()

# Print the Fuzzy Partition Coefficient (FPC)
print(f"Fuzzy Partition Coefficient (FPC): {fpc:.3f}")
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
    doc.add_heading('Performance of Fuzzy C-Means Clustering:', level=2)
    doc.add_paragraph(
        '- Fuzzy C-Means (FCM) is effective for clustering datasets where data points can belong to multiple clusters with varying degrees of membership.\n'
        '- It works well for datasets with overlapping clusters and provides more flexibility than traditional clustering methods like K-Means.\n'
        '- The algorithm has a time complexity of O(n^2) in the worst case, making it less efficient for very large datasets.'
    )

    doc.add_heading('Drawbacks of Fuzzy C-Means Clustering:', level=2)
    doc.add_paragraph(
        '- Scalability: Due to its high time complexity, FCM is not suitable for very large datasets.\n'
        '- Sensitivity to Initialization: The performance of the algorithm can be highly dependent on the initial membership values.\n'
        '- Choice of Fuzziness Parameter: The choice of the fuzziness parameter (m) can significantly impact the results, and there is no universal best choice.'
    )

    doc.add_heading('Remedies:', level=2)
    doc.add_paragraph(
        '- For large datasets, consider using more scalable clustering algorithms like K-Means or DBSCAN.\n'
        '- Experiment with different initialization methods to improve the performance of the algorithm.\n'
        '- Use techniques like cross-validation to determine the optimal value of the fuzziness parameter (m).'
    )

    # Save the document
    doc.save('Assignment_9_Fuzzy_CMeans.docx')

# Run the function to generate the document
create_assignment_docx()