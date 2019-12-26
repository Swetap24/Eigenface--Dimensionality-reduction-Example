# Eigenface--Dimensionality-reduction-Example


Face recognition is an important task in computer vision and machine learning. This code provides a classical approach called Eigenfaces. I have used face images from the Yale Face Database B, which contains face images from 10 people under 64 lighting conditions.

# Dataset 
It contains three sets of variables:
• image: each element is a face image (50 × 50 matrix). Used matplotlib.pyplot.imshow to visualize the image. The data is stored in a cell array.
• personID: each element is the ID of the person, which takes values from 1 to 10.
• subsetID: each element is the ID of the subset which takes values from 1 to 5. Here the face
images are divided into 5 subsets. Each subset contains face images from all people, but with
different lighting conditions.

# PCA 
PCA function takes the data matrix (each row being a sample) and target dimensionality d (lower than or equal to the original
dimensionality) as the input, and outputs the eigenvectors.These eigenvectors are called eigenfaces (when displayed as
images).
