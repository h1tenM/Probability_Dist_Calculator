# utils.py (updated)
import numpy as np
import plotly.graph_objects as go
from scipy import stats

def get_inequality_symbol(prob_type):
    """Helper function to get inequality symbol for display"""
    symbols = {
        'less': '<',
        'less_equal': '≤',
        'greater': '>',
        'greater_equal': '≥'
    }
    return symbols.get(prob_type, '<')

def plot_distribution(dist_name, params, x_range, calc_type='pdf', x_value=None, prob_type='less'):
    """
    Create a Plotly figure for a probability distribution with proper highlighting.
    
    Args:
        dist_name (str): Name of the distribution ('normal', 'exponential', 'binomial', 'poisson')
        params (dict): Distribution parameters
        x_range (tuple): Range for x-axis (min, max)
        calc_type (str): 'pdf' or 'cdf'
        x_value (float/int, optional): Specific x value for probability calculation
        prob_type (str): Type of probability calculation ('less', 'less_equal', 'greater', 'greater_equal')
    
    Returns:
        dict: Dictionary containing plot data for frontend
    """
    # Determine if distribution is discrete or continuous
    is_discrete = dist_name in ['binomial', 'poisson']
    
    # Generate x values
    if is_discrete:
        x = np.arange(int(x_range[0]), int(x_range[1]) + 1)
        if len(x) == 0:
            x = np.array([0])
    else:
        x = np.linspace(x_range[0], x_range[1], 500)
    
    # Get distribution object
    if dist_name == 'normal':
        dist = stats.norm(loc=params['mu'], scale=params['sigma'])
    elif dist_name == 'exponential':
        dist = stats.expon(scale=1/params['lambda'])
    elif dist_name == 'binomial':
        dist = stats.binom(n=params['n'], p=params['p'])
    elif dist_name == 'poisson':
        dist = stats.poisson(mu=params['lambda'])
    else:
        raise ValueError("Unsupported distribution name")
    
    # Calculate y values
    if calc_type == 'pdf':
        y = dist.pmf(x) if is_discrete else dist.pdf(x)
    else:
        y = dist.cdf(x)
    
    # Prepare graph data
    graph_data = {
        'type': 'discrete' if is_discrete else 'continuous',
        'x': x.tolist(),
        'y': y.tolist(),
        'name': f'{dist_name.title()} {calc_type.upper()}',
        'y_title': 'Probability Density' if calc_type == 'pdf' else 'Cumulative Probability'
    }
    
    # Add highlighting if x_value is provided
    if x_value is not None:
        graph_data['x_value'] = float(x_value)
        graph_data['prob_type'] = prob_type
        graph_data['prob_label'] = f'P(X {get_inequality_symbol(prob_type)} {x_value:.2f})'
        
        if is_discrete:
            # Discrete case - highlight bars
            bar_colors = []
            
            if prob_type in ['less', 'less_equal']:
                compare_val = x_value if prob_type == 'less_equal' else x_value - 1
                highlight_indices = np.where(x <= compare_val)[0]
            else:  # greater or greater_equal
                compare_val = x_value if prob_type == 'greater_equal' else x_value + 1
                highlight_indices = np.where(x >= compare_val)[0]
            
            # Create color array
            bar_colors = ['red' if i in highlight_indices else 'royalblue' for i in range(len(x))]
            graph_data['bar_colors'] = bar_colors
        else:
            # Continuous case - fill area under curve
            if prob_type in ['less', 'less_equal']:
                fill_mask = x <= x_value
            else:  # greater or greater_equal
                fill_mask = x >= x_value
            
            graph_data['fill_x'] = x[fill_mask].tolist()
            graph_data['fill_y'] = y[fill_mask].tolist()
    
    return graph_data

def calculate_binomial_probabilities(n, p, x):
    """
    Calculate all binomial probabilities for given parameters.
    Args:
        n (int): Number of trials
        p (float): Probability of success
        x (int): Number of successes
    Returns:
        dict: Dictionary containing all probabilities and statistics
    """
    dist = stats.binom(n=n, p=p)
    # Calculate individual probabilities
    prob_exact = dist.pmf(x)
    prob_less = dist.cdf(x-1)
    prob_less_equal = dist.cdf(x)
    prob_greater = 1 - dist.cdf(x)
    prob_greater_equal = 1 - dist.cdf(x-1)
    # Calculate mean and standard deviation
    mean = n * p
    std = np.sqrt(n * p * (1-p))
    # Calculate probability distribution table
    x_vals = np.arange(n+1)
    pmf_vals = dist.pmf(x_vals)
    cdf_vals = dist.cdf(x_vals)
    distribution_table = [
        {'x': int(x), 'pmf': f'{p:.4f}', 'cdf': f'{c:.4f}'}
        for x, p, c in zip(x_vals, pmf_vals, cdf_vals)
    ]
    return {
        'prob_exact': prob_exact,
        'prob_less': prob_less,
        'prob_less_equal': prob_less_equal,
        'prob_greater': prob_greater,
        'prob_greater_equal': prob_greater_equal,
        'mean': mean,
        'std': std,
        'distribution_table': distribution_table
    }