import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#function to create excel file
def create_file(file_name):
    with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
        for i in range(4):
            data = np.round(np.random.uniform(-1, 1, 15), 2)
            df = pd.DataFrame(data, columns=['Values'])
            df.to_excel(writer, sheet_name=f'Sheet{i+1}', index=False)


#function to calculate mean and deviation 
def calculate_stats(file_name):
    df = pd.read_excel(file_name, sheet_name=None)

    stats = {}
    for sheet_name, data in df.items(): 
        mean = data['Values'].mean()
        std_dev = data['Values'].std()
        stats[sheet_name] = (mean, std_dev)
    return stats

#function to plot distribution
def plot_distributions(stats, directory='.'):
    x = np.linspace(-2, 2, 1000)
    
    plt.figure(figsize=(10, 6))
    
    for sheet_name, (mean, std_dev) in stats.items():
        y = norm.pdf(x, mean, std_dev)
        plt.plot(x, y, label=f'{sheet_name}: Mean ={mean:.2f}, S.D. ={std_dev:.2f}')

    plt.title('Dataset Normal Distribution Plot')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(f"{directory}/Figure.png")
    # plt.clf()

#Main function
def main():
    file_name = 'dataset.xlsx'
    create_file(file_name)
    
    stats = calculate_stats(file_name)
    plot_distributions(stats)

if __name__ == "__main__":
    main()