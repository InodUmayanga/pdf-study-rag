"""Source text for the evaluation fixture PDF.

Original study-notes-style content written for this repository. Each page
covers one topic and contains facts that appear on no other page, so every
evaluation question has exactly one correct page.
"""

PAGES = [
    (
        "Page 1 - Vectors and the Dot Product",
        "A vector is an ordered list of numbers. In these notes vectors are "
        "written as rows, for example v = (1, 2, 3). The number of entries "
        "is called the dimension, so v is a three-dimensional vector.\n\n"
        "The dot product of two vectors of the same dimension multiplies "
        "matching entries and adds the results. Worked example: the dot "
        "product of (1, 2, 3) and (4, 5, 6) is 1*4 + 2*5 + 3*6 = 32.\n\n"
        "The length (also called the norm) of a vector is the square root "
        "of its dot product with itself. Worked example: the length of "
        "(3, 4) is the square root of 9 + 16, which is 5.\n\n"
        "Two vectors are orthogonal when their dot product is zero. "
        "Worked example: (1, 2) and (2, -1) are orthogonal because "
        "1*2 + 2*(-1) = 0.",
    ),
    (
        "Page 2 - Matrices and Matrix Multiplication",
        "A matrix is a rectangular grid of numbers with a number of rows "
        "and a number of columns. A matrix with 2 rows and 3 columns is "
        "called a 2 by 3 matrix and holds six entries.\n\n"
        "Two matrices can be multiplied only when the number of columns "
        "of the first equals the number of rows of the second. The result "
        "has the rows of the first and the columns of the second, so a "
        "2 by 3 matrix times a 3 by 4 matrix gives a 2 by 4 matrix.\n\n"
        "Matrix multiplication is not commutative: A times B is usually "
        "different from B times A. It is associative, so (A times B) times "
        "C equals A times (B times C).\n\n"
        "The identity matrix has ones on the main diagonal and zeros "
        "elsewhere. Multiplying any matrix by an identity matrix of the "
        "right size leaves it unchanged.",
    ),
    (
        "Page 3 - Determinants and Inverses",
        "The determinant of a 2 by 2 matrix with entries a, b in the first "
        "row and c, d in the second row is a*d - b*c. Worked example: the "
        "matrix with rows (2, 3) and (1, 4) has determinant 2*4 - 3*1 = 5."
        "\n\n"
        "A square matrix has an inverse exactly when its determinant is "
        "not zero. Such a matrix is called invertible or non-singular. A "
        "matrix with determinant zero is called singular and has no "
        "inverse.\n\n"
        "The inverse of a 2 by 2 matrix is found by swapping a and d, "
        "changing the signs of b and c, and dividing every entry by the "
        "determinant.\n\n"
        "The determinant of a product equals the product of the "
        "determinants: det(A times B) = det(A) times det(B).",
    ),
    (
        "Page 4 - Eigenvalues and Eigenvectors",
        "An eigenvector of a square matrix A is a non-zero vector v such "
        "that A times v equals lambda times v for some number lambda. That "
        "number lambda is the eigenvalue paired with v.\n\n"
        "Eigenvalues are found by solving the characteristic equation "
        "det(A - lambda I) = 0, where I is the identity matrix. For an "
        "n by n matrix this equation has degree n in lambda.\n\n"
        "Worked example: the diagonal matrix with rows (5, 0) and (0, 7) "
        "has eigenvalues 5 and 7, and its eigenvectors are the standard "
        "basis vectors (1, 0) and (0, 1).\n\n"
        "The sum of the eigenvalues of a matrix equals its trace, which is "
        "the sum of the entries on the main diagonal. The product of the "
        "eigenvalues equals the determinant.",
    ),
    (
        "Page 5 - Probability Basics and Bayes Rule",
        "A sample space is the set of all possible outcomes of an "
        "experiment. An event is a subset of the sample space. The "
        "probability of an event is a number between 0 and 1, and the "
        "probabilities of all outcomes add up to 1.\n\n"
        "The conditional probability of A given B is P(A and B) divided "
        "by P(B), written P(A | B). Two events are independent when "
        "P(A and B) = P(A) times P(B).\n\n"
        "Bayes rule turns one conditional probability into the other: "
        "P(A | B) = P(B | A) times P(A) divided by P(B).\n\n"
        "Worked example: a fair six-sided die is rolled once. The "
        "probability of an even number given that the result is greater "
        "than 3 is 2/3, because the outcomes 4, 5 and 6 remain and two of "
        "them are even.",
    ),
    (
        "Page 6 - Random Variables, Expectation and Variance",
        "A random variable assigns a number to each outcome of an "
        "experiment. A discrete random variable takes countably many "
        "values, such as the number of heads in ten coin flips.\n\n"
        "The expectation (also called the expected value or mean) of a "
        "discrete random variable is the sum of each value multiplied by "
        "its probability. Worked example: a fair six-sided die has "
        "expectation (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5.\n\n"
        "The variance measures spread. It is the expectation of the "
        "squared distance from the mean. The standard deviation is the "
        "square root of the variance and has the same units as the "
        "variable.\n\n"
        "Expectation is linear: the expectation of a sum of random "
        "variables is the sum of their expectations, whether or not they "
        "are independent.",
    ),
    (
        "Page 7 - Common Distributions",
        "A Bernoulli random variable takes the value 1 with probability p "
        "and 0 with probability 1 - p. Its mean is p and its variance is "
        "p times (1 - p).\n\n"
        "A binomial random variable counts the number of successes in n "
        "independent Bernoulli trials with the same p. Its mean is n times "
        "p. Worked example: 20 flips of a fair coin have a binomial "
        "distribution with mean 10.\n\n"
        "The normal distribution is the bell-shaped curve described by its "
        "mean mu and standard deviation sigma. About 68 percent of its "
        "probability lies within one standard deviation of the mean and "
        "about 95 percent within two standard deviations.\n\n"
        "The uniform distribution on an interval gives equal probability "
        "density to every point in the interval; its mean is the midpoint "
        "of the interval.",
    ),
]
