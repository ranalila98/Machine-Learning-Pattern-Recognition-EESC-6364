import numpy as np
data = np.array([-7.82, -6.68, 4.36, 6.72, -8.64, -6.87, 4.47, 6.73, -7.71, -6.91, 
                 6.18, 6.72, -6.25, -6.94, 8.09, 6.81, -5.19, -6.38, 4.08, 6.27])

#Question A-> Known: P(ω1)=0.5, σ1 = σ2 = 1; Unknown: µ1 and µ2.
def case_a(data):
    mu1, mu2 = np.random.rand(2) * 10  # Random initialization of µ1 and µ2.
    converged = False
    P_w1 = 0.5 
    sigma1 = sigma2 = 1  
    
    while not converged:
        p_w1 = P_w1 * (1 / np.sqrt(2 * np.pi * sigma1**2)) * np.exp(-0.5 * ((data - mu1) / sigma1)**2)
        p_w2 = (1 - P_w1) * (1 / np.sqrt(2 * np.pi * sigma2**2)) * np.exp(-0.5 * ((data - mu2) / sigma2)**2)
        total_p = p_w1 + p_w2
        
        # Normalize the probabilities
        r1 = p_w1 / total_p  
        r2 = p_w2 / total_p  
        
        mu1_new = np.sum(r1 * data) / np.sum(r1)  
        mu2_new = np.sum(r2 * data) / np.sum(r2) 
        
        if np.allclose([mu1, mu2], [mu1_new, mu2_new], atol=1e-6):
            converged = True        
        mu1, mu2 = mu1_new, mu2_new  # Update the means for the next iteration
        
    return mu1, mu2

#Question B-> Known: P(ω1)=0.5; Unknown: σ1 = σ2 = σ, µ1 and µ2
def case_b(data):
    mu1, mu2, sigma = np.random.rand(3) * 10
    converged = False
        
    while not converged:
        p_w1 = 0.5 * (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((data - mu1) / sigma)**2)
        p_w2 = 0.5 * (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((data - mu2) / sigma)**2)
        total_p = p_w1 + p_w2
        p_w1 /= total_p
        p_w2 /= total_p
        
        
        mu1_new = np.sum(p_w1 * data) / np.sum(p_w1)
        mu2_new = np.sum(p_w2 * data) / np.sum(p_w2)
        sigma_new = np.sqrt(np.sum(p_w1 * (data - mu1_new)**2) + np.sum(p_w2 * (data - mu2_new)**2)) / len(data)
        
        if np.allclose([mu1, mu2, sigma], [mu1_new, mu2_new, sigma_new], atol=1e-6):
                converged = True
            
        mu1, mu2, sigma = mu1_new, mu2_new, sigma_new
    
    return mu1, mu2, sigma

# Question C-> Known: P(ω1)=0.5; Unknown: σ1, σ2, µ1 and µ2.
def case_c(data):
    # Known P(ω1) = 0.5; Unknown: σ1, σ2, μ1, μ2
    mu1, mu2, sigma1, sigma2 = np.random.rand(4) * 10
    converged = False
    
    while not converged:
        p_w1 = 0.5 * (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-0.5 * ((data - mu1) / sigma1)**2)
        p_w2 = 0.5 * (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-0.5 * ((data - mu2) / sigma2)**2)
        total_p = p_w1 + p_w2
        
        p_w1 /= total_p
        p_w2 /= total_p
        
        mu1_new = np.sum(p_w1 * data) / np.sum(p_w1)
        mu2_new = np.sum(p_w2 * data) / np.sum(p_w2)
        sigma1_new = np.sqrt(np.sum(p_w1 * (data - mu1_new)**2) / np.sum(p_w1))
        sigma2_new = np.sqrt(np.sum(p_w2 * (data - mu2_new)**2) / np.sum(p_w2))
        
        if np.allclose([mu1, mu2, sigma1, sigma2], [mu1_new, mu2_new, sigma1_new, sigma2_new], atol=1e-6):
            converged = True         
        mu1, mu2, sigma1, sigma2 = mu1_new, mu2_new, sigma1_new, sigma2_new
    
    return mu1, mu2, sigma1, sigma2

# Question D-> Unknown: P(ω1), σ1, σ2, µ1 and µ2.
def case_d(data):
    # Initialize parameters with random values
    mu1, mu2, sigma1, sigma2 = np.random.rand(4) * 10
    p_w1 = np.random.rand()  # Random initialization for P(ω1)
    converged = False

    while not converged:
        # E-step: calculate responsibilities
        p_w1_val = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-0.5 * ((data - mu1) / sigma1) ** 2)
        p_w2_val = (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-0.5 * ((data - mu2) / sigma2) ** 2)
        total_p = p_w1_val + p_w2_val
        
        # Avoid division by zero
        total_p = np.where(total_p == 0, 1e-10, total_p)  
        
        p_w1 = p_w1_val / total_p
        p_w2 = p_w2_val / total_p
        
        mu1_new = np.sum(p_w1 * data) / np.sum(p_w1)
        mu2_new = np.sum(p_w2 * data) / np.sum(p_w2)
        sigma1_new = np.sqrt(np.sum(p_w1 * (data - mu1_new) ** 2) / np.sum(p_w1))
        sigma2_new = np.sqrt(np.sum(p_w2 * (data - mu2_new) ** 2) / np.sum(p_w2))
        
        # Update prior probability
        p_w1_new = np.sum(p_w1) / len(data) 

        if np.allclose([mu1, mu2, sigma1, sigma2, p_w1.mean()], 
                       [mu1_new, mu2_new, sigma1_new, sigma2_new, p_w1_new], 
                       atol=1e-6):
            converged = True   
                           
        # Update all parameters for the next iteration
        mu1, mu2, sigma1, sigma2, p_w1 = mu1_new, mu2_new, sigma1_new, sigma2_new, p_w1_new
    
    return mu1, mu2, sigma1, sigma2, p_w1

def main():
    print(f"Sample Data X1:{data}")    
    
    mu1_a, mu2_a = case_a(data)
    print(f"Question A: Mean 1 = {mu1_a}, Mean 2 = {mu2_a}")
    
    mu1_b, mu2_b, sigma_b = case_b(data)
    print(f"Question B: Mean 1 = {mu1_b}, Mean 2 = {mu2_b}, sigma = {sigma_b}")
    
    mu1_c, mu2_c, sigma1_c, sigma2_c = case_c(data)
    print(f"Question C: Mean 1 = {mu1_c}, Mean 2 = {mu2_c}, sigma 1 = {sigma1_c}, sigma 2 = {sigma2_c}")
    
    mu1_d, mu2_d, sigma1_d, sigma2_d, p_w1_d = case_d(data)
    print(f"Question D: Mean 1 = {mu1_d}, Mean 2 = {mu2_d}, sigma 1 = {sigma1_d}, sigma 2 = {sigma2_d}, P(ω1) = {p_w1_d}")


if __name__ == "__main__":
    main()