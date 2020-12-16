import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
# set the seed so we get the same values for theta
np.random.seed(50)


# loading the data into a dataframe.
Salary_df = pd.read_csv('SalaryData.csv')
Salary_df

# Define constants.
Prior_Means = np.array([[40000],[0], [0], [0]])

# eye creates an identity 4x4 matrix.
Prior_Cov = np.eye(4) * 100
measurement_sigma = 250
# predictions of y's.
y_hats = np.zeros((30,1))


# Loop thorugh each observation to update the priors thetas.
for index, row in Salary_df.iterrows():
    t = row['YearsExperience']
    y = row['Salary']
    H_k = np.array([[1, t, t**2, t**3]])
    current_Means = Prior_Means
    current_Covariances = Prior_Cov
    """current_Cov and Means is respectively P_k-1 and m_k-1,
    H_k is the current data point. Operator @ 
    is matrix multiplication and .T is transpose. """
    Prior_Cov = current_Covariances - current_Covariances @ H_k.T \
    @ (H_k @ current_Covariances @ H_k.T + measurement_sigma**2)**(-1) \
    @  (H_k @ current_Covariances)
    Prior_Means = Prior_Cov @ (H_k.T * measurement_sigma**(-2) * y  + \
        inv(current_Covariances) @ current_Means)
    # after 10 observations stop.
    # if index==9: break
# final m_k, P_k matrices.
print(Prior_Cov)
print(Prior_Means)

"""firstly we need to derive the thetas from a 
multivariate normal distribtuion with the new m_k, 
P_k and then predict y's hat based on random thetas 
drawn from the multivarite normal distribution"""
thetas = np.random.multivariate_normal(Prior_Means.reshape(4,), Prior_Cov)
# for each example we predict the y_hat.
for index, row in Salary_df.iterrows():
    t = row['YearsExperience']
    y_hats[index] = thetas[0] + thetas[1] * t + \
    thetas[2] * t**2 + thetas[3] * t**3

#plot the fitted curve on the data.

plt.plot(Salary_df['YearsExperience'], Salary_df['Salary'], 'o', color = 'black')
plt.plot(Salary_df['YearsExperience'], y_hats, color = 'm' )
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.savefig('Fitted_Curve.png')
